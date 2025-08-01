import matplotlib.pyplot as plt
import seaborn as sns
import platform
import schnetpack as spk
import schnetpack.transform as trn
import torch
import numpy as np
import pytorch_lightning as pl
import os
import spainn

from schnetpack.datasets import MD17
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig

from ground_state_md.setup_logger import setup_logger

logger = setup_logger("debug")


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

def get_bin_number(data: np.ndarray) -> int:
    """ 
    Get number of bins for a histogram using Scott's rule.
    Args:
        data (np.ndarray): Array of values.

    Returns:
        int: Number of bins for histogram.
    """
    # Calculate bin width
    bin_width = 3.5 * np.std(data) / len(data) ** (1/3)
    num_bins = int((data.max() - data.min()) / bin_width)
    logger.debug(f"Calculated number of bins: {num_bins} for data with shape {data.shape}")
    return num_bins

####################################################################################

def load_md17_dataset(data_prefix: str, molecule: str = 'ethanol', dataset_name: str = "rMD17",
              pin_memory: bool | None = None, num_workers: int | None = None) -> spk.data.datamodule.AtomsDataModule | MD17:
    """
    Load the MD17 dataset for the specified molecule.
    Args:
        data_prefix (str): Path to the dataset.
        molecule (str): Name of the molecule to load. Default is 'ethanol'.
        dataset_name (str): Name of the dataset. Default is 'rMD17'.
        pin_memory (bool): Whether to use pinned memory. Default is None.
        num_workers (int): Number of workers for data loading. Default is None.
    Returns:
        spk.data.datamodule.AtomsDataModule | MD17: The loaded MD17 dataset.
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

####################################################################################

def load_xtb_dataset(db_path: str, num_workers: int, batch_size: int, transforms: list, split_file: str | None = None, pin_memory: bool | None = None) -> spk.data.datamodule.AtomsDataModule:
    """
    Load an XTB dataset from the specified path. 
    Note: data.prepare_data() and data.setup() do not need to be called here, since they will be called by pl.trainer.fit().
    Args:
        db_path (str): Path to the dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Batch size for the dataset.
        transforms (list): List of transforms to apply to the dataset.
        split_file (str | None): Path to the split file. Default is None.
        pin_memory (bool | None): Whether to use pinned memory. Default is None.
    Returns:
        spk.data.datamodule.AtomsDataModule: The loaded XTB dataset.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if num_workers == -1:
        num_workers = 0 if platform.system() == "Darwin" else 31
    else:
        num_workers = num_workers
    logger.debug(f"pin_memory: {pin_memory}")
    logger.debug(f"num_workers: {num_workers}")

    # load xtb dataset with subclass of pl.LightningDataModule
    data = spk.data.AtomsDataModule(
        db_path,
        batch_size=batch_size,
        distance_unit='Bohr',
        property_units={'energy':'Hartree', 'forces':'Hartree/Bohr'},
        split_file=split_file,
        transforms=transforms,
        num_workers=num_workers,
        pin_memory=pin_memory, # set to false, when not using a GPU
    )
    data.prepare_data()
    data.setup()
    logger.info(f"loaded xtb dataset: {data}")

    if not isinstance(data, pl.LightningDataModule):
        raise ValueError("The loaded dataset is not an instance of pl.LightningDataModule. Please check the dataset path and configuration.")

    return data

def load_xtb_dataset_without_config(db_path: str, batch_size: int, split_file: str | None = None, pin_memory: bool | None = None, num_workers: int = -1) -> pl.LightningDataModule:
    """
    Load anXTB dataset from the specified path.
    Args:
        db_path (str): Path to the dataset.
        batch_size (int): Batch size for the dataset.
        split_file (str | None): Path to the split file. Default is None.
        pin_memory (bool | None): Whether to use pinned memory. Default is None.
        num_workers (int): Number of workers for data loading. Default is -1.
    Returns:
        pl.LightningDataModule: The loaded XTB dataset.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if num_workers == -1:
        num_workers = 0 if platform.system() == "Darwin" else 31 
        logger.debug(f"num_workers: {num_workers}")

    # load xtb dataset
    data = spk.data.AtomsDataModule(
        db_path,
        batch_size=batch_size,
        distance_unit='Bohr',
        property_units={'energy':'Hartree', 'forces':'Hartree/Bohr'},
        split_file=split_file,
        transforms=[],
        num_workers=num_workers,
        pin_memory=pin_memory, # set to false, when not using a GPU
    )
    
    data.prepare_data()
    data.setup()

    logger.info(f"loaded xtb dataset: {data}")

    return data


def load_xtb_dataset_without_given_splits(db_path: str, batch_size: int = 10, pin_memory: bool | None = None, num_workers: int | None = None) -> pl.LightningDataModule:
    """
    Load anXTB dataset from the specified path.
    Args:
        db_path (str): Path to the dataset.
        batch_size (int): Batch size for the dataset. Default is 10.
        pin_memory (bool | None): Whether to use pinned memory. Default is None.
        num_workers (int | None): Number of workers for data loading. Default is None.
    Returns:
        pl.LightningDataModule: The loaded XTB dataset.
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
        distance_unit='Bohr',
        property_units={'energy':'Hartree', 'forces':'Hartree/Bohr'},
        num_train=1000,
        num_val=1000,
        transforms=[],
        num_workers=num_workers,
        pin_memory=pin_memory, # set to false, when not using a GPU
    )
    
    data.prepare_data()
    data.setup()

    logger.info(f"loaded xtb dataset: {data}")

    return data

####################################################################################

def load_config(config_path: str, config_name: str, job_name: str) -> DictConfig:
    """
    Load the configuration from the specified path and name.
    Args:
        config_path (str): Path to the configuration directory.
        config_name (str): Name of the configuration file.
        job_name (str): Name of the job for Hydra.
    Returns:
        DictConfig: Loaded configuration.
    """
    with initialize(config_path=config_path, job_name=job_name, version_base="1.1"):
        return compose(config_name=config_name)

def get_split_path(data_prefix: str, trajectory_dir: str, fold: int = 0) -> str:
    """
    Get the path to the split file for the given trajectory directory and fold.
    Args:
        data_prefix (str): The prefix path to the data directory.
        trajectory_dir (str): The directory containing the trajectory data.
        fold (int): The fold number for cross-validation (default: 0).
    Returns:
        str: The path to the split file.
    """
    split_file = os.path.join(data_prefix, "splits", trajectory_dir, f"inner_splits_{fold}.npz")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Missing split file: {split_file}")
    logger.debug(f"Split file: {split_file}")
    return split_file

def setup_datamodule(data_cfg: DictConfig, datapath: str, split_file: str) -> spk.data.AtomsDataModule | spainn.SPAINN:
    """
    Setup the data module for the given configuration.
    Args:
        data_cfg (DictConfig): Configuration for the data module.
        datapath (str): Path to the data.
        split_file (str): Path to the split file.
    Returns:
        spk.data.AtomsDataModule | spainn.SPAINN: The instantiated data module for usage with SchNetPack (ground state) or SPaiNN (exited state).
    """
    dm = instantiate(data_cfg, datapath=datapath, split_file=split_file)
    dm.prepare_data()
    dm.setup()
    logger.info(f"Loaded datamodule: {dm}")
    return dm

def get_num_workers(num_workers: int) -> int:
    """
    Get the number of workers for data loading.
    Args:
        num_workers (int): Number of workers specified by the user, if -1, it will be set to 0 on macOS and 8 on Linux.
    Returns:
        int: Number of workers to use for data loading.
    """
    if num_workers != -1:
        return num_workers
    else:
        return 0 if platform.system() == 'Darwin' else 8
    
def set_data_units_in_config(cfg_org_data: DictConfig, ase_units: dict) -> DictConfig:
    """ 
    Set the ASE units in the configuration based on the distance and energy units specified in the configuration.
    Args:
        cfg_org_data (DictConfig): The original configuration containing the distance and energy units.
        ase_units (dict): A dictionary containing the ASE units for distance, energy, and forces.
    Returns:
        DictConfig: The updated configuration with ASE units set.
    """
    # Set the ASE units in the configuration
    cfg_org_data.distance_unit = ase_units['distance']
    cfg_org_data.property_units.energy = ase_units['energy']
    cfg_org_data.property_units.forces = ase_units['forces']
    return cfg_org_data