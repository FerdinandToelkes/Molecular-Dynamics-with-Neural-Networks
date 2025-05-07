import matplotlib.pyplot as plt
import seaborn as sns
import platform
import schnetpack as spk
import schnetpack.transform as trn
import torch

from schnetpack.datasets import MD17
from schnetpack.data import ASEAtomsData

from md_with_schnet.setup_logger import setup_logger

logger = setup_logger(logging_level_str="debug")


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

    logger.debug("Finished setting up plotting configuration:")
    logger.debug(f"fontsize: {fontsize}")
    logger.debug(f"aspect_ratio: {aspect_ratio}")
    logger.debug(f"width_fraction: {width_fraction}")

def set_data_prefix() -> str:
    """
    Set the data prefix depending on the system.

    Returns:
        str: Data prefix path.
    """
    logger.debug("Setting data prefix")
    logger.debug(f"System: {platform.system()}")
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


def load_xtb_dataset(db_path: str, batch_size: int, split_file: any = None, pin_memory: any = None, num_workers: int = -1) -> ASEAtomsData:
    """
    Load anXTB dataset from the specified path.
    Args:
        db_path (str): Path to the dataset.
        batch_size (int): Batch size for the dataset.
        split_file (any): Path to the split file. Default is None.
        pin_memory (bool): Whether to use pinned memory. Default is None.
        num_workers (int): Number of workers for data loading. Default is None.
    Returns:
        ASEAtomsData: The loaded XTB dataset.
    """
    logger.debug(f"num_workers when entering: {num_workers}")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if num_workers == -1:
        num_workers = 0 if platform.system() == "Darwin" else 31 
        logger.debug(f"num_workers: {num_workers}")

    # load xtb dataset
    data = spk.data.AtomsDataModule(
        db_path,
        batch_size=batch_size,
        distance_unit='Ang',
        property_units={'energy':'Hartree', 'forces':'Hartree/Bohr'},
        split_file=split_file,
        transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ],
        num_workers=num_workers,
        pin_memory=pin_memory, # set to false, when not using a GPU
    )
    
    data.prepare_data()
    data.setup()

    logger.info(f"loaded xtb dataset: {data}")

    return data

def load_xtb_dataset_without_given_splits(db_path: str, batch_size: int = 10, pin_memory: bool = None, num_workers: int = None) -> ASEAtomsData:
    """
    Load anXTB dataset from the specified path.
    Args:
        db_path (str): Path to the dataset.
        batch_size (int): Batch size for the dataset. Default is 10.
        pin_memory (bool): Whether to use pinned memory. Default is None.
        num_workers (int): Number of workers for data loading. Default is None.
    Returns:
        ASEAtomsData: The loaded XTB dataset.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if num_workers is None:
        num_workers = 0 if platform.system() == "Darwin" else 31 
        logger.debug(f"num_workers: {num_workers}")

    # load xtb dataset
    data = spk.data.AtomsDataModule(
        db_path,
        batch_size=batch_size,
        distance_unit='Ang',
        property_units={'energy':'Hartree', 'forces':'Hartree/Bohr'},
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
    
    data.prepare_data()
    data.setup()

    logger.info(f"loaded xtb dataset: {data}")

    return data