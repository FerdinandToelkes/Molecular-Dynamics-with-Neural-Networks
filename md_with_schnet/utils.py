import logging
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import schnetpack as spk
import schnetpack.transform as trn
import torch

from schnetpack.datasets import MD17

logger = logging.getLogger(__name__)
if (logger.hasHandlers()):
    logger.handlers.clear()

def setup_logger(logging_level: int = logging.INFO, external_level: int = logging.WARNING) -> logging.Logger:
    """ Set up a logger for the module.
    Args:
        logging_level (int, optional): Logging level. Defaults to logging.INFO.
        external_level (int, optional): Logging level for external libraries like SchNetPack. Defaults to logging.WARNING.
    Returns:
        logging.Logger: Configured logger.
    """
    # Clear any existing handlers (added by other libraries like schnetpack)
    logger = logging.getLogger(__name__)

    # Don't propagate to root logger to avoid duplicate messages
    # which are I think caused by the schnetpack logger (bad practice)
    logger.propagate = False

    logger.setLevel(logging_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - line %(lineno)d - %(message)s')

    # Create console handler and set level to debug
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Reduce noise from schnetpack and possibly other libraries
    for external_module in ["schnetpack"]:
        logging.getLogger(external_module).setLevel(external_level)

    return logger


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

def set_data_prefix() -> str:
    """
    Set the data prefix depending on the system.

    Returns:
        str: Data prefix path.
    """
    if platform.system() == 'Darwin':
        return '/Users/ferdinandtolkes/whk/data'
    elif platform.system() == 'Linux':
        return '/loctmp/tof54964/data'
    else:
        raise ValueError('Unknown system. Please set data_prefix manually.')


def load_md17_dataset(data_prefix: str, molecule: str = 'ethanol', dataset_name: str = "rMD17",
              pin_memory: bool = None, num_workers: int = None) -> MD17:
    """
    Load the MD17 dataset for the specified molecule.
    Args:
        data_prefix (str): Path to the dataset.
        molecule (str): Name of the molecule to load. Default is 'ethanol'.
        dataset_name (str): Name of the dataset. Default is 'rMD17'.
        pin_memory (bool): Whether to use pinned memory. Default is None.
        num_workers (int): Number of workers for data loading. Default is None.
    Returns:
        MD17: The loaded MD17 dataset.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if num_workers is None:
        num_workers = 0 if platform.system() == "Darwin" else 1 
    
    db_path = f'{data_prefix}/{dataset_name}/db_data/{molecule}.db'

    if dataset_name == "rMD17":
        data = spk.data.AtomsDataModule(
            db_path,
            batch_size=10,
            distance_unit='Ang',
            property_units={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'},
            num_train=1000,
            num_val=1000,
            transforms=[
                trn.ASENeighborList(cutoff=5.),
                trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
                trn.CastTo32()
            ],
            num_workers=num_workers,
            pin_memory=pin_memory, # set to false, when not using a GPU
        )
    elif dataset_name == "MD17":
        data = MD17(
            db_path,
            molecule=molecule,
            batch_size=10,
            num_train=1000,
            num_val=1000,
            transforms=[
                trn.ASENeighborList(cutoff=5.),
                trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
                trn.CastTo32()
            ],
            num_workers=num_workers,
            pin_memory=pin_memory, # set to false, when not using a GPU
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Use 'rMD17' or 'MD17'.")

    
    data.prepare_data()
    data.setup()

    return data

