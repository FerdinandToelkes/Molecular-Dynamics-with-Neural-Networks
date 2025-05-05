import os
import numpy as np
import logging

from ase import Atoms
from schnetpack.data import ASEAtomsData
from io import StringIO

from md_with_schnet.utils import set_data_prefix

# Create a logger
level = "DEBUG"
external_level = "WARNING"
logger = logging.getLogger(__name__)
logger.setLevel(level)

# Don't propagate to root logger to avoid duplicate messages
# which are I think caused by the schnetpack logger (bad practice)
logger.propagate = False

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create console handler and set level to debug
stream_handler = logging.StreamHandler()
stream_handler.setLevel(level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Reduce noise from schnetpack and possibly other libraries
for external_module in ["schnetpack"]:
    logging.getLogger(external_module).setLevel(external_level)

# Example command to run the script from within code directory:
"""
python -m md_with_schnet.data_analysis.prepare_xtb
"""

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
        data_energies = data["old_energies"][sort_order]
        data_forces = data["old_forces"][sort_order]
    else:
        data_coords = data["coords"]
        data_energies = data["old_energies"]
        data_forces = data["old_forces"]

    return data_coords, data_energies, data_forces

def extract_data_from_xyz(path: str, extra_lines: int, number_of_atoms: any = None) -> tuple:
    """
    Extract data from a .xyz file and return the coordinates.
    Args:
        path (str): Path to the .xyz file.
        extra_lines (int): Number of extra lines in each coordinates block, e.g., header lines.
    Returns:
        np.ndarray: Array of coordinates for different time steps.
    """
    # Read the file line by line
    with open(path, 'r') as file:
        lines = file.readlines()
        if number_of_atoms is None:
            # If number_of_atoms is not provided, read it from the first line
            number_of_atoms = int(lines[0].strip())

    # Parameters
    logger.debug(f'number_of_atoms: {number_of_atoms}')
    frame_size = number_of_atoms + extra_lines  # Total lines per frame

    # Extract atom data lines
    atom_lines = []
    for i in range(0, len(lines), frame_size):
        # Each frame consists of n header lines and the atom data lines
        atom_data = lines[i + extra_lines:i + extra_lines + number_of_atoms]
        atom_lines.extend(atom_data)

    # Convert the atom data lines into a NumPy array
    atom_data_str = ''.join(atom_lines)
    data = np.loadtxt(StringIO(atom_data_str), usecols=(1, 2, 3))
    # Reshape the trajectory data
    data = data.reshape(-1, number_of_atoms, 3)  # Shape: (Nframes, Natoms, 3)
    logger.debug(f'data.shape: {data.shape}')
    return data, number_of_atoms

def main():
    # setup
    data_prefix = set_data_prefix() + "/turbo_test"
    # path = data_prefix + f'MOTOR_MD_XTB/T300_1/traj.xyz'
    # target_path = data_prefix + f'MOTOR_MD_XTB/T300_1/traj.db'

    path = data_prefix + f'turbo_test/traj_1.xyz'
    target_path = data_prefix + f'turbo_test/traj_1.db'

    logger.debug("extracting trajectory data from xyz file")
    traj, number_of_atoms = extract_data_from_xyz(path=f'{data_prefix}/traj_1.xyz', extra_lines=2,)
    E = np.loadtxt(f'{data_prefix}/energies_1.txt', usecols=(3))    # total energy
    logger.debug(f'E.shape: {E.shape}')
    print(f'E header: {E[:5]}')

    logger.debug("extracting gradients data from txt file")
    grads = extract_data_from_xyz(path=f'{data_prefix}/grads_1.txt', extra_lines=1, number_of_atoms=number_of_atoms)

    exit()
    
    numbers = data["nuclear_charges"]
    atoms_list = []
    property_list = []
    for positions, energies, forces in zip(data_coords, data_energies, data_forces):
        ats = Atoms(positions=positions, numbers=numbers)
        # convert energies to array if it is not already
        if not isinstance(energies, np.ndarray):
            energies = np.array([energies]) # compare with shape of data within the tutorial

        properties = {'energy': energies, 'forces': forces}
        property_list.append(properties)
        atoms_list.append(ats)

    print('Properties:', property_list[0])

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
    print('Number of reference calculations:', len(new_dataset))
    print('Available properties:')

    for p in new_dataset.available_properties:
        print('-', p)
    print()

    print(f"new_dataset: {new_dataset}")
    example = new_dataset[0]
    print('Properties of molecule with id 0:')

    for k, v in example.items():
        print('-', k, ':', v.shape)

if __name__=="__main__":
    
    main()