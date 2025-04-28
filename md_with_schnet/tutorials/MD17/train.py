import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
from schnetpack.datasets import MD17


from tutorials.utils import set_data_prefix

# Example command to run the script from within schnetpack directory:
"""
python -m tutorials.MD17.train
"""

def load_md17(data_prefix: str, output_path: str, molecule: str = 'ethanol') -> MD17:
    """
    Load the MD17 dataset for the specified molecule.
    Args:
        data_prefix (str): Path to the dataset.
        output_path (str): Path to save the output.
        molecule (str): Name of the molecule to load. Default is 'ethanol'.
    Returns:
        MD17: The loaded MD17 dataset.
    """
    data = MD17(
        f'{data_prefix}/{molecule}.db',
        molecule=molecule,
        batch_size=10,
        num_train=1000,
        num_val=1000,
        transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ],
        num_workers=1,
        split_file=os.path.join(output_path, "split.npz"),
        pin_memory=False, # set to false, when not using a GPU
    )
    data.prepare_data()
    data.setup()
    return data


if __name__=="__main__":
    forcetut = os.path.expanduser('~/whk/code/schnetpack/tutorials/MD17/output')
    os.makedirs(forcetut, exist_ok=True)

    data_prefix = set_data_prefix()
    # Load data from MD17
    ethanol_data = load_md17(data_prefix, forcetut, molecule='ethanol')
    print("Done setting up data")

    # Check if force and energy are in the dataset
    properties = ethanol_data.dataset[0]
    print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])

    # Check if the shape of the data is correct
    print('Forces:\n', properties[MD17.forces])
    print('Shape:\n', properties[MD17.forces].shape)

    # Build the model
    cutoff = 5.
    n_atom_basis = 30

    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    # One output module for energy and one for forces
    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=MD17.energy)
    pred_forces = spk.atomistic.Forces(energy_key=MD17.energy, force_key=MD17.forces)

    # 
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets(MD17.energy, add_mean=True, add_atomrefs=False)
        ]
    )

    # setup the loss function
    output_energy = spk.task.ModelOutput(
        name=MD17.energy,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.01,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name=MD17.forces,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.99,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    # setup the task
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )

    # setup logging and checkpointing
    logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(forcetut, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]
    # train the model for 5 epochs
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=forcetut,
        max_epochs=5, # for testing, we restrict the number of epochs
    )
    trainer.fit(task, datamodule=ethanol_data)
