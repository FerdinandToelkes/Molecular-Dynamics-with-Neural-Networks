import torch
import logging
import argparse
import yaml
import schnetpack as spk
import schnetpack.transform as trn
import os
import platform
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader
from tqdm import tqdm
from schnetpack.datasets import MD17

from utils import set_plotting_config
from data_analysis.utils import set_data_prefix
from data_analysis.molecule_analyzer import MoleculeTrajectoryComparer


# Example command to run the script from within schnetpack directory:
"""
python -m data_analysis.main
"""

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO, 
    format="%(name)s - %(asctime)s - %(levelname)s - %(message)s"
)

# Get the logger for this script (one logger per module)
logger = logging.getLogger(__name__)

def parse_args() -> dict:
    """ Parse command-line arguments. 

    Returns:
        dict: Dictionary containing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Plotting script for MD17 dataset")
    parser.add_argument("--molecule_name", type=str, default="ethanol", help="Name of the molecule to load (default: ethanol)")
    parser.add_argument("--sorted", action="store_true", help="Use the sorted version of the revised molecule (default: False)")
    parser.add_argument("--show_plots", action="store_true", help="Show plots before saving them (default: False)")
    return vars(parser.parse_args())


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
            num_val=100,
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
            num_val=100,
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

def plot_comparisons(plot_dir: str, plot_type: str, comparer_function: callable, extra_args: dict = {}, show_plots: bool = False):
    """ General function to create subplot grids and call comparison functions. 
    
    Args:
        plot_dir (str): Directory to save the plot to.
        plot_type (str): Type of plot to create.
        comparer_function (function): Function to compare values.
        extra_args (dict, optional): Additional arguments for the comparer function. Defaults to {}.
    """
    sharey = "autocorrelation" in plot_type or "distributions" in plot_type
    fig, axes = plt.subplots(3, 2, sharey=sharey)  
    axes = axes.reshape(-1, 2)  # Ensure each row has 2 subplots
    
    for i, key in enumerate(["energies", "total_forces", "displacements"]):
        comparer_function(key, axes=axes[i], **extra_args.get(key, {}))

    plt.tight_layout()
    path = os.path.join(plot_dir, f"{plot_type}.pdf")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to: {path}")
    if show_plots:
        plt.show()
    plt.close()

def main(molecule_name: str, sorted: bool, show_plots: bool):
    # setup
    output_dir = os.path.expanduser('~/whk/code/schnetpack/data_analysis/output')
    os.makedirs(output_dir, exist_ok=True)
    data_prefix = set_data_prefix()

    # load original and revised MD17 dataset (e.g., Ethanol)
    if sorted:
        rmd17 = load_md17_dataset(data_prefix, molecule=f"{molecule_name}_sorted", dataset_name="rMD17")
    else:
        rmd17 = load_md17_dataset(data_prefix, molecule=molecule_name, dataset_name="rMD17")

    md17 = load_md17_dataset(data_prefix, molecule=molecule_name, dataset_name="MD17")
    logger.info(f"loaded datasets: {md17} and {rmd17}")
    
    # setup plotting use textwidth and height to set aspect ratio
    # Text width: 468.0pt, Text height: 665.5pt
    set_plotting_config(fontsize=10, aspect_ratio=468/525, width_fraction=1)
 
    # Load the YAML configuration
    with open("data_analysis/plot_configs.yaml", "r") as file:
        plot_config = yaml.safe_load(file)

    for nr_configs_as_str in plot_config.keys():
        start = time.time()
        n_samples = int(nr_configs_as_str.split("_")[0])
        c = plot_config[nr_configs_as_str]
        logger.info(f"Plotting for n_samples: {n_samples}")

        plot_dir = f"plots/MD17_vs_rMD17_sorted={sorted}/{molecule_name}/{nr_configs_as_str}"
        os.makedirs(plot_dir, exist_ok=True)

        batch_size = 1000 if n_samples > 1000 else n_samples

        org_loader = DataLoader(
            dataset=md17.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,   # or more depending on your CPU
            pin_memory=False
        )
   
        revised_loader = DataLoader(
            dataset=rmd17.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,   # or more depending on your CPU
            pin_memory=False
        )
        
        
        dataloaders = {"MD17": org_loader, "rMD17": revised_loader}

        # Initialize the trajectory comparer
        desired_batches = n_samples // batch_size
        comparer = MoleculeTrajectoryComparer(dataloaders, desired_batches, plot_dir=plot_dir)
        
        if c["plot_distributions"]:
            plot_comparisons(
                plot_dir,
                "distributions",
                comparer.plot_distribution_comparison,
                extra_args={
                    "energies": {"xlabel": "Energy (kcal/mol)", "set_title": True, "nr_of_xticks": 5}, 
                    "total_forces": {"xlabel": r"Force Magnitude ($\mathrm{eV}/\mathrm{\AA}$)"},
                    "displacements": {"xlabel": r"Displacement ($\mathrm{\AA}$)", "legend_location": "upper left"}
                    },
                show_plots=show_plots
            )

        if c["plot_values"]:
            plot_comparisons(
                plot_dir,
                "values",
                comparer.plot_values_comparison,
                extra_args={
                    "energies": {"ylabel": "Energy (kcal/mol)", "set_title": True}, 
                    "total_forces": {"ylabel": r"Force Magnitude ($\mathrm{eV}/\mathrm{\AA}$)"},
                    "displacements": {"ylabel": r"Displacement ($\mathrm{\AA}$)","set_xlabel": True}
                    },
                show_plots=show_plots
            )

        if c["plot_values_connected"]:
            plot_comparisons(
                plot_dir,
                "values_connected",
                comparer.plot_values_connected_comparison,
                extra_args={
                    "energies": {"ylabel": "Energy (kcal/mol)", "set_title": True}, 
                    "total_forces": {"ylabel": r"Force Magnitude ($\mathrm{eV}/\mathrm{\AA}$)"},
                    "displacements": {"ylabel": r"Displacement ($\mathrm{\AA}$)", "set_xlabel": True}
                    },
                show_plots=show_plots
            )

        if c["plot_autocorrelation"]:
            lags = c["autocorrelation_lags"]
            plot_comparisons(
                plot_dir,
                "autocorrelation",
                comparer.plot_autocorrelation_comparison,
                extra_args={
                    "energies": {"set_title": True, "lags": lags},
                    "total_forces": {"lags": lags},
                    "displacements": {"set_xlabel": True, "lags": lags},
                    },
                show_plots=show_plots
            )
        logger.info(f"Time taken for {n_samples} samples: {time.time() - start:.2f} seconds")


if __name__=="__main__":
    args = parse_args()
    main(**args)