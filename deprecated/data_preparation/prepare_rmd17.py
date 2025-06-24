import os
import numpy as np
import argparse

from schnetpack.data import ASEAtomsData

from md_with_schnet.setup_logger import setup_logger
from md_with_schnet.utils import set_data_prefix
from md_with_schnet.preprocessing.prepare_xtb_in_atomic_units import convert_trajectory_to_ase, get_overview_of_dataset


logger = setup_logger(logging_level_str="debug")

# Example command to run the script from within code directory:
"""
python -m md_with_schnet.preprocessing.prepare_rmd17 --molecule_name ethanol --sort_configs
"""

def parse_args() -> dict:
    """ 
    Parse command-line arguments.
    Returns:
        dict: Dictionary containing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare rMD17 data for usage in SchNetPack")
    parser.add_argument("--molecule_name", type=str, default="ethanol", help="Name of the molecule to load (default: ethanol)")
    parser.add_argument("--sort_configs", action="store_true", help="Whether to sort the data points with the old indices or not (default: False)")
    return vars(parser.parse_args())

def extract_data_from_npz(data: np.ndarray, sort_configs: bool = True) -> tuple:
    """
    Extract data from a .npz file and return the coordinates, energies, and forces.

    Args:
        data (np.ndarray): The data loaded from the .npz file.
        sort_configs (bool): Whether to sort configurations by old_indices.

    Returns:
        tuple: Tuple containing coordinates, energies, and forces.
    """
    if sort_configs:
        # sort the configurations by old_indices
        old_indices = data["old_indices"]
        sort_order = np.argsort(old_indices)

        data_coords = data["coords"][sort_order]
        data_energies = data["energies"][sort_order]
        data_forces = data["forces"][sort_order]
    else:
        data_coords = data["coords"]
        data_energies = data["energies"]
        data_forces = data["forces"]

    return data_coords, data_energies, data_forces

def main(molecule_name: str, sort_configs: bool):
    """
    Main function to prepare the rMD17 dataset for usage in SchNetPack.
    Args:
        molecule_name (str): Name of the molecule to load.
        sort_configs (bool): Whether to sort the data points with the old indices or not.
    """
    data_prefix = set_data_prefix()
    path = os.path.join(data_prefix, f'rMD17/npz_data/{molecule_name}.npz')
    if sort_configs:
        target_path = os.path.join(data_prefix, f'rMD17/db_data/{molecule_name}_sorted.db')
    else:
        target_path = os.path.join(data_prefix, f'rMD17/db_data/{molecule_name}.db')
    logger.debug(f"Loading data from {path}")
    logger.debug(f"Saving data to {target_path}")

    # extract data from the .npz file
    data = np.load(path)
    coords_traj, energy_traj, forces_traj = extract_data_from_npz(data, sort_configs)
    atomic_numbers = data["nuclear_charges"]

    atoms_list, property_list = convert_trajectory_to_ase(
        coords_traj=coords_traj,
        energy_traj=energy_traj,
        forces_traj=forces_traj,
        atomic_numbers=atomic_numbers
    )

    # Create a new dataset in the schnetpack format
    if os.path.exists(target_path):
        print(f"File {target_path} already exists, loading it.")
        new_dataset = ASEAtomsData(target_path)
    else:
        print(f"File {target_path} does not exist, creating it.")
        # create a new dataset
        new_dataset = ASEAtomsData.create(
            target_path,
            distance_unit='Ang',
            property_unit_dict={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'}
        )
        # add systems to the dataset
        new_dataset.add_systems(property_list, atoms_list)

    # get overview of the dataset
    get_overview_of_dataset(new_dataset)

if __name__=="__main__":
    args = parse_args()
    main(**args)