import os
import numpy as np

from ase import Atoms
from schnetpack.data import ASEAtomsData

from md_with_schnet.utils import set_data_prefix


# Example command to run the script from within code directory:
"""
python -m md_with_schnet.data_analysis.prepare_rmd17
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

def main():
    sort_configs = True
    molecule_name = 'uracil'
    data_prefix = set_data_prefix()
    path = data_prefix + f'rMD17/npz_data/{molecule_name}.npz'
    if sort_configs:
        target_path = data_prefix + f'rMD17/db_data/{molecule_name}_sorted.db'
    else:
        target_path = data_prefix + f'rMD17/db_data/{molecule_name}.db'

    # extract data from the .npz file
    data = np.load(path)
    data_coords, data_energies, data_forces = extract_data_from_npz(data, sort_configs=sort_configs)
    
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