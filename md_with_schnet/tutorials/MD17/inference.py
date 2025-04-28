import torch
import os
import schnetpack as spk
import schnetpack.transform as trn

from schnetpack.datasets import MD17
from ase import Atoms
from ase import io


from tutorials.utils import set_data_prefix
from tutorials.MD17.train import load_md17

# Example command to run the script from within schnetpack directory:
"""
python -m tutorials.MD17.inference
"""


if __name__=="__main__":
    # Load data from MD17
    forcetut = os.path.expanduser('~/whk/code/schnetpack/tutorials/MD17/output')
    data_prefix = set_data_prefix()
    ethanol_data = load_md17(data_prefix, forcetut, molecule='ethanol')
    print("Done setting up data")

    # set device
    device = torch.device("cuda")

    # load model
    model_path = os.path.join(forcetut, "best_inference_model")
    best_model = torch.load(model_path, map_location=device)

    # set up converter
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )

    # create atoms object from dataset
    structure = ethanol_data.test_dataset[0]
    atoms = Atoms(
        numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
    )

    # convert atoms to SchNetPack inputs and perform prediction
    inputs = converter(atoms)
    results = best_model(inputs)

    print(results)

    # Change units of the results
    calculator = spk.interfaces.SpkCalculator(
        model_file=model_path,
        neighbor_list=trn.ASENeighborList(cutoff=5.0),
        energy_key=MD17.energy,
        force_key=MD17.forces,
        energy_unit="kcal/mol",
        position_unit="Ang",
    )

    atoms.set_calculator(calculator)

    print("Prediction:")
    print("energy:", atoms.get_total_energy())
    print("forces:", atoms.get_forces())

    # Generate a directory for the ASE computations
    ase_dir = os.path.join(forcetut, 'ase_calcs')
    os.makedirs(ase_dir, exist_ok=True)

    # Write a sample molecule
    molecule_path = os.path.join(ase_dir, 'ethanol.xyz')
    io.write(molecule_path, atoms, format='xyz')

    # Load the model with ASE interface
    ethanol_ase = spk.interfaces.AseInterface(
        molecule_path,
        ase_dir,
        model_file=model_path,
        neighbor_list=trn.ASENeighborList(cutoff=5.0),
        energy_key=MD17.energy,
        force_key=MD17.forces,
        energy_unit="kcal/mol",
        position_unit="Ang",
        device="cpu",
        dtype=torch.float64,
    )

    # geometry optimization
    ethanol_ase.optimize(fmax=1e-2)

    # normal mode analysis
    ethanol_ase.compute_normal_modes()