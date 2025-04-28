import argparse
import os

from torch_geometric.datasets import MD17
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator

from data_preparation.main import extract_data

def parse_args() -> dict:
    """ Parse command line arguments.
    
    Returns:
        dict: Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description='Convert npz to xyz')
    parser.add_argument('-in', '--in_name', type=str, default='ethanol_from_npz', help='Name of the target file')
    args = parser.parse_args()
    return dict(vars(args))


def main(in_name: str):
    """ Convert npz to xyz format.
    
    Args:
        in_name (str): Name of the target file.
    """
    # set output filename and remove if it already exists
    out_filename = f"{in_name}.xyz"
    if os.path.exists(out_filename):
        print(f"Removing existing file: {out_filename}")
        os.remove(out_filename)

    # read in npz file
    data = MD17(root="data/MD17", name=in_name)
    _, symbols, positions, energies, forces = extract_data(data)

    # iterate over data and write continuously to extxyz file
    for idx in range(len(positions)):
        curr_atoms = Atoms(
            # set atomic positions
            positions=positions[idx],
            # set chemical symbols / species
            symbols=symbols[idx], 
            # molecules in vacuum -> no pbc
            pbc=False
        )

        # set calculator to assign targets (nothing is calculated here)
        calculator = SinglePointCalculator(curr_atoms, energy=energies[idx], forces=forces[idx])
        curr_atoms.calc = calculator
        
        write(out_filename, curr_atoms, format='xyz', append=True)


if __name__ == "__main__":
    args = parse_args()
    main(**args)