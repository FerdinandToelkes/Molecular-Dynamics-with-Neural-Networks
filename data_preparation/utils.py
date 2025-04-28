import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

from torch_geometric.datasets import MD17

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO, 
    format="%(name)s - %(asctime)s - %(levelname)s - %(message)s"
)

# Get the logger for this script (one logger per module)
logger = logging.getLogger(__name__)


def set_plotting_config(fontsize: int = 10, aspect_ratio: float = 1.618, width_fraction: float = 1.0, text_usetex: bool = True,
                        latex_text_width_in_pt: int = 468):
    """ Set global plotting configuration for Matplotlib and Seaborn. 
    
    Args:   
        fontsize (int, optional): Font size for text elements. Defaults to 10.
        aspect_ratio (float, optional): Aspect ratio of the figure. Defaults to 1.618.
        width_fraction (float, optional): Fraction of the text width to use for the figure width in latex. Defaults to 1.0.
        text_usetex (bool, optional): Use LaTeX for text rendering. Defaults to True.
        latex_text_width_in_pt (int, optional): LaTeX text width in points. Defaults to 468 (from Physical Review B).
    """
    latex_text_width_in_in = width_fraction * latex_text_width_in_pt / 72  # Convert pt to inches
    scale_factor = width_fraction + 0.25  if width_fraction < 1.0 else 1.0


    # Set Matplotlib rcParams
    plt.rcParams.update({
        "font.family": "serif" if text_usetex else "sans-serif",
        "text.usetex": text_usetex,
        'font.size': fontsize * scale_factor,  
        'text.latex.preamble': r'\usepackage{lmodern}',
        "axes.labelsize": fontsize * scale_factor,
        "axes.titlesize": fontsize * scale_factor,
        "xtick.labelsize": (fontsize - 2) * scale_factor,
        "ytick.labelsize": (fontsize - 2) * scale_factor,
        "legend.fontsize": (fontsize - 2) * scale_factor,
        "axes.linewidth": 0.8 * scale_factor,
        "lines.linewidth": 0.8 * scale_factor,
        "grid.linewidth": 0.6 * scale_factor,
        'lines.markersize': 5 * width_fraction,
        "figure.autolayout": True,
        "figure.figsize": (latex_text_width_in_in, latex_text_width_in_in / aspect_ratio),
    }) 

    # Set color palette
    sns.set_palette("colorblind")

def extract_data_from_MD17(dataset: MD17) -> tuple:
    """ Extract atomic positions, forces, and energies from dataset.
    
    Args:
        dataset (MD17): PyG dataset object.

    Returns:
        tuple: Tuple containing number of atoms, atomic symbols, atomic positions, energies, and forces.
    """
    logger.debug(f"Extracting data from MD17 dataset.")
    nr_atoms = dataset[0].pos.shape[0]
    positions = dataset.pos.numpy().reshape(-1, nr_atoms, 3)
    symbols = dataset.z.numpy().reshape(-1, nr_atoms)
    energies = dataset.energy.numpy()
    forces = dataset.force.numpy().reshape(-1, nr_atoms, 3)
    logger.info(f"Number of atoms: {nr_atoms}")
    logger.info(f"positions.shape: {positions.shape}")
    logger.info(f"symbols.shape: {symbols.shape}")
    logger.info(f"energies.shape: {energies.shape}")
    logger.info(f"forces.shape: {forces.shape}")
    exit()
    return nr_atoms, symbols, positions, energies, forces

def extract_energies(dataset: MD17) -> np.ndarray:
    """ Extract energies from dataset.
    
    Args:
        dataset (MD17): PyG dataset object.

    Returns:
        np.ndarray: Energies for each molecular configuration
    """
    logger.debug(f"Extracting energies from MD17 dataset.")
    energies = dataset.energy.numpy()
    logger.debug(f"energies.shape: {energies.shape}")
    return energies

def get_bin_number(data) -> int:
    """ Get number of bins for energy histogram using Scott's rule.
    
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