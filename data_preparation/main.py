import os
import argparse
import yaml
import logging
import matplotlib.pyplot as plt

from torch_geometric.datasets import MD17

from data_preparation.utils import set_plotting_config
from data_preparation.molecule_analyzer import MoleculeTrajectoryComparer

# example usage: python3.10 -m data_preparation.main --molecule_name "ethanol" 

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
    parser.add_argument("--show_plots", action="store_true", help="Show plots before saving them (default: False)")
    return vars(parser.parse_args())

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

# TODO:
# - why is there no autocorrelation for rMD17?
# - Why is the energy changing seamingly uncontiously? -> is there shuffeling happening?
# - Check if energy is sampled from same distribution as in original MD17
# - Check if forces are sampled from same distribution as in original MD17
# - Compute correlation between energy and forces
# - Compute correlation between energy and positions
# - Compute correlation between forces and positions
# - gather differences in latex

def main(molecule_name: str, show_plots: bool):
    """ Main function to load and analyze the MD17 dataset.
    
    Args:
        molecule_name (str): Name of the molecule to load.
    """
    # load original and revised MD17 dataset (e.g., Ethanol)
    md17 = MD17(root="data/MD17", name=molecule_name)
    rmd17 = MD17(root="data/MD17", name=f"revised {molecule_name}")
    logger.info(f"loaded datasets: {md17} and {rmd17}")
    
    # setup plotting use textwidth and height to set aspect ratio
    # Text width: 468.0pt, Text height: 665.5pt
    set_plotting_config(fontsize=10, aspect_ratio=468/525, width_fraction=1)
 
    # Load the YAML configuration
    with open("data_preparation/plot_configs.yaml", "r") as file:
        plot_config = yaml.safe_load(file)

    for nr_configs_as_str in plot_config.keys():
        n_samples = int(nr_configs_as_str.split("_")[0])
        c = plot_config[nr_configs_as_str]
        logger.info(f"Plotting for n_samples: {n_samples}")

        plot_dir = f"plots/MD17_vs_rMD17/{molecule_name}/{nr_configs_as_str}"
        os.makedirs(plot_dir, exist_ok=True)

        # take the last n_samples from the datasets -> equilibrium
        org_dataset = md17[:n_samples]
        revised_dataset = rmd17[:n_samples]
        datasets = {"MD17": org_dataset, "rMD17": revised_dataset}

        # Initialize the trajectory comparer
        comparer = MoleculeTrajectoryComparer(datasets, plot_dir=plot_dir)
        
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
        

        

    
    

if __name__ == "__main__":
    args = parse_args()
    main(**args)
    





