import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import logging
import argparse

from schnetpack.datasets import MD17

from ground_state_md.utils import setup_logger, load_md17_dataset, set_data_prefix


logger = setup_logger(logging.INFO)

def parse_args() -> dict:
    """ Parse command-line arguments. 

    Returns:
        dict: Dictionary containing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for training SchNetPack on MD17 dataset")
    parser.add_argument("--dataset_name", type=str, default="rMD17", help="Name of the dataset to load from (default: rMD17)")
    parser.add_argument("--molecule_name", type=str, default="ethanol", help="Name of the molecule to load (default: ethanol)")
    return vars(parser.parse_args())



def main(dataset_name: str, molecule_name: str):
    # setup
    output_dir = os.path.expanduser('~/whk/code/schnetpack/data_analysis/output')
    os.makedirs(output_dir, exist_ok=True)
    data_prefix = set_data_prefix()
    output_dir = f'{data_prefix}/output'
    print(f"output_dir: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load original and revised MD17 dataset (e.g., Ethanol)
    data = load_md17_dataset(data_prefix, molecule=molecule_name, dataset_name=dataset_name)
    logger.info(f"loaded dataset: {data}")

    # print some information about the dataset
    properties = data.dataset[0]
    properties_str = ''.join(f'{i}\n' for i in properties.keys())
    logger.info(f"Loaded properties:\n{properties_str}")
    logger.info(f'Forces:\n{properties["forces"]}')
    logger.info(f'Shape:\n{properties["forces"].shape}')

    # BUILD MODEL
    # Define representation
    cutoff = 5.
    n_atom_basis = 30

    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    # Define output modules
    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=MD17.energy)
    pred_forces = spk.atomistic.Forces(energy_key=MD17.energy, force_key=MD17.forces)

    # Combine into a model
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets(MD17.energy, add_mean=True, add_atomrefs=False)
        ]
    )

    # Define loss function
    output_energy = spk.task.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.01,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name="forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.99,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    # Define task
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )

    # set logger and callbacks
    train_logger = pl.loggers.TensorBoardLogger(save_dir=output_dir)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(output_dir, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    # set up the trainer and fit the model
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=train_logger,
        default_root_dir=output_dir,
        max_epochs=5, # for testing, we restrict the number of epochs
    )
    trainer.fit(task, datamodule=data)


if __name__ == "__main__":
    args = parse_args()
    main(**args)