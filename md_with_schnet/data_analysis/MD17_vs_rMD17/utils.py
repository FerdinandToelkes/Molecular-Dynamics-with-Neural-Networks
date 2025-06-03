import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from schnetpack.datasets import MD17

from md_with_schnet.setup_logger import setup_logger

logger = setup_logger(logging_level_str="info")


def extract_data_from_MD17(data: MD17, desired_batches: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract atomic data from a SchNetPack MD17 dataset.

    Args:
        data (MD17): SchNetPack dataset (not PyG).
        desired_batches (int): Number of batches to extract from the dataset.

    Returns:
        Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - Number of atoms
            - Atomic numbers (n_samples, n_atoms)
            - Positions (n_samples, n_atoms, 3)
            - Energies (n_samples,)
            - Forces (n_samples, n_atoms, 3)
    """
    logger.debug(f"Extracting data from MD17 dataset.")
    
    # initialize lists to store data
    positions_list = []
    symbols_list = []
    energies_list = []
    forces_list = []

    for i, batch in enumerate(tqdm(data, total=desired_batches, desc="Loading data")):
        if i >= desired_batches:
            break

        # Append data to lists
        positions_list.append(batch["_positions"].numpy())
        symbols_list.append(batch["_atomic_numbers"].numpy())
        energies_list.append(batch["energy"].numpy())
        forces_list.append(batch["forces"].numpy())

    # Concatenate the lists into numpy arrays
    positions = np.concatenate(positions_list, axis=0)
    symbols = np.concatenate(symbols_list, axis=0)
    energies = np.concatenate(energies_list, axis=0)
    forces = np.concatenate(forces_list, axis=0)
    
    # Reshape the data
    nr_atoms = positions.shape[1]
    positions = positions.reshape(-1, nr_atoms, 3)
    symbols = symbols.reshape(-1, nr_atoms)
    forces = forces.reshape(-1, nr_atoms, 3)
    
    # Log the shapes of the data
    logger.debug(f"Number of atoms: {nr_atoms}")
    logger.debug(f"positions.shape: {positions.shape}")
    logger.debug(f"symbols.shape: {symbols.shape}")
    logger.debug(f"energies.shape: {energies.shape}")
    logger.debug(f"forces.shape: {forces.shape}")
    return nr_atoms, symbols, positions, energies, forces


def extract_data_from_MD17_fast(data: MD17) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract atomic data from a SchNetPack MD17 dataset.

    Args:
        data (MD17): SchNetPack dataset (not PyG).

    Returns:
        Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - Number of atoms
            - Atomic numbers (n_samples, n_atoms)
            - Positions (n_samples, n_atoms, 3)
            - Energies (n_samples,)
            - Forces (n_samples, n_atoms, 3)
    """
    logger.debug(f"Extracting data from MD17 dataset.")
    nr_atoms = data[0]["_positions"].shape[0]
    batch_size = len(data)

    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,  # adjust depending on memory
        shuffle=False,
        num_workers=0,   # or more depending on your CPU
        pin_memory=False
    )

    positions_list = []
    symbols_list = []
    energies_list = []
    forces_list = []

    for batch in loader:
        positions_list.append(batch["_positions"].numpy())
        symbols_list.append(batch["_atomic_numbers"].numpy())
        energies_list.append(batch["energy"].numpy())
        forces_list.append(batch["forces"].numpy())

    positions = np.concatenate(positions_list, axis=0)
    symbols = np.concatenate(symbols_list, axis=0)
    energies = np.concatenate(energies_list, axis=0)
    forces = np.concatenate(forces_list, axis=0)

    logger.debug(f"Number of atoms: {nr_atoms}")
    logger.debug(f"positions.shape: {positions.shape}")
    logger.debug(f"symbols.shape: {symbols.shape}")
    logger.debug(f"energies.shape: {energies.shape}")
    logger.debug(f"forces.shape: {forces.shape}")
    return nr_atoms, symbols, positions, energies, forces


def get_bin_number(data) -> int:
    """ 
    Get number of bins for energy histogram using Scott's rule.
    
    Returns:
        int: Number of bins for histogram.
    """
    # Calculate bin width
    bin_width = 3.5 * np.std(data) / len(data) ** (1/3)
    num_bins = int((data.max() - data.min()) / bin_width)
    return num_bins

def get_save_suffix(molecule_name: str, n_samples: int) -> str:
    """ Get suffix to append to saved plots.

    Args:
        molecule_name (str): Name of the molecule.
        n_samples (int): Number of samples loaded from the dataset. 

    Returns:
        str: Suffix to append to saved plots.
    """
    save_suffix = molecule_name.split(" ")[-1]
    if 'revised' in molecule_name:
        save_suffix = save_suffix + "_rMD17"
    else:
        save_suffix = save_suffix + "_MD17"
    save_suffix = f"{save_suffix}_n={n_samples}"
    return save_suffix