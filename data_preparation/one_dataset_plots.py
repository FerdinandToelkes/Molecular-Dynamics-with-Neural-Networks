import os
import argparse
import yaml

from torch_geometric.datasets import MD17

from data_preparation.utils import set_plotting_config, get_save_suffix
from data_preparation.molecule_analyzer import MoleculeTrajectoryAnalyzer

# example usage: python3.10 -m data_preparation.one_dataset_plots --molecule_name "revised ethanol" --n_samples 10_000

def parse_args() -> dict:
    """ Parse command-line arguments. 

    Returns:
        dict: Dictionary containing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Plotting script for MD17 dataset")
    parser.add_argument("--molecule_name", type=str, default="revised ethanol", help="Name of the molecule to load (default: revised ethanol)")
    parser.add_argument("--n_samples", type=int, default=10_000, help="Number of samples to load from the dataset (default: 10_000)")
    return vars(parser.parse_args())


def main(molecule_name: str, n_samples: int):
    """ Main function to load and analyze the MD17 dataset.
    
    Args:
        molecule_name (str): Name of the molecule to load.
        n_samples (int): Number of samples to load from the dataset.
    """
    # get save suffix
    save_suffix = "_" + get_save_suffix(molecule_name, n_samples)
    
    # load revised MD17 dataset (e.g., Ethanol)
    org_dataset = MD17(root="data/MD17", name=molecule_name)
    print(f"loaded dataset: {org_dataset}")
    org_dataset = org_dataset[:n_samples]

    # setup plotting
    set_plotting_config(fontsize=8, aspect_ratio=4/3, width_fraction=0.5)
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    molecule_analyzer = MoleculeTrajectoryAnalyzer(org_dataset, save_suffix, plot_dir)
 
    # distribution plots
    molecule_analyzer.plot_distribution("energies", "Energy (kcal/mol)", f"energy_distribution")
    
    molecule_analyzer.plot_distribution("total_forces", r"Force Magnitude ($\mathrm{eV}/\mathrm{\AA}$)", f"force_distribution")
    molecule_analyzer.plot_distribution("displacements", r"Displacement ($\mathrm{\AA}$)", f"displacement_distribution")

    # value plots
    molecule_analyzer.plot_values("energies", "Energy (kcal/mol)", f"energy_values")
    molecule_analyzer.plot_values("total_forces", r"Force Magnitude ($\mathrm{eV}/\mathrm{\AA}$)", f"force_values")
    molecule_analyzer.plot_values("displacements", r"Displacement ($\mathrm{\AA}$)", f"displacement_values")

    
    # autocorrelation plots
    molecule_analyzer.plot_autocorrelation("energies", f"energy_autocorrelation")
    molecule_analyzer.plot_autocorrelation("total_forces", f"force_autocorrelation")
    molecule_analyzer.plot_autocorrelation("displacements", f"displacement_autocorrelation")

    


    
    

if __name__ == "__main__":
    args = parse_args()
    main(**args)
    





