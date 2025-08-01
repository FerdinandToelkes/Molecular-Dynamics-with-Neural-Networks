import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.ticker import MaxNLocator

from ground_state_md.data_analysis.MD17_vs_rMD17.utils import extract_data_from_MD17, get_bin_number



# Define colors for the plots
MAIN_COLOR = "#1f77b4"  # Deep Blue (Main line)
MEAN_COLOR = "#d62728"  # Crimson Red (Mean)
STD_COLOR = "#ff7f0e"   # Burnt Orange (Std deviation)
GRID_COLOR = "#bfbfbf"  # Soft Gray (Gridlines)

class MoleculeTrajectoryAnalyzer:
    """ Class to analyze and plot molecular data from datasets in SchNetPack compatible format (.db loaded with ASE). """
    def __init__(self, dataloader: DataLoader, dataset_name: str, desired_batches: int, plot_dir: str = "plots"):
        """ Initialize the Plotter object.

        Args:
            dataloader (Dataloader): Dataset object containing molecular data.
            dataset_name (str): Name of the dataset.
            desired_batches (int): Number of batches to load from the dataset.
            plot_dir (str, optional): Directory to save plots. Defaults to "plots".
        """
        self.dataset_name = dataset_name
        self.desired_batches = desired_batches
        self.plot_dir = plot_dir
        self.data = self.get_data(dataloader)
        

    def get_data(self, dataloader: DataLoader) -> dict:
        """ Helper function to get data from a single dataset. """
        nr_atoms, symbols, positions, energies, forces = extract_data_from_MD17(dataloader, self.desired_batches)
        print(f"Extracted data from dataset")
        total_forces = np.mean(np.linalg.norm(forces, axis=2), axis=1)
        displacements = np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=2), axis=1)
        return {
            "nr_atoms": nr_atoms,
            "symbols": symbols,
            "positions": positions,
            "energies": energies,
            "forces": forces,
            "total_forces": total_forces,
            "displacements": displacements
        }

    def plot_distribution(self, data_key: str, ax: plt.axes, xlabel: str, set_title: bool = False, legend_location: str = "upper right"):
        """ 
        General function to plot a histogram with mean and standard deviation lines.
        Args:
            data_key (str): Key to access data from the data dictionary.
            ax (plt.Axes): Matplotlib axis object.
            xlabel (str): Label for the x-axis.    
            set_title (bool, optional): Set title for the plot. Defaults to False.
            legend_location (str, optional): Location of the legend. Defaults to "upper right". 
        """
        data = self.data[data_key]
        num_bins = get_bin_number(data)
        self.plot_histogram_with_mean_std(ax, data, num_bins)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        if set_title:
            ax.set_title(self.dataset_name)
        ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR) 
        ax.legend(frameon=True, loc=legend_location)

    def plot_histogram_with_mean_std(self, ax: plt.Axes, data: np.ndarray, num_bins: int):
        """ Plot a histogram with mean and standard deviation lines.

        Args:
            ax (plt.Axes): Matplotlib axis object.
            data (np.ndarray): Data to plot.
            num_bins (int): Number of bins for the histogram.
        """
        counts, bins, _ = ax.hist(data, bins=num_bins, color=MAIN_COLOR, alpha=0.6, edgecolor="none", linewidth=0.6)

        max_height = max(counts)
        mean, std = data.mean(), data.std()
        ax.vlines(mean, 0, max_height * 1.05, color=MEAN_COLOR, linestyle="--", linewidth=1.2, label="Mean")
        ax.vlines([mean - std, mean + std], 0, max_height * 1.05, color=STD_COLOR, linestyle="--", linewidth=1.2, label=r"Mean $\pm$ Std.")

    def plot_values(self, data_key: str, ax: plt.axes, ylabel: str, set_title: bool = False, set_xlabel: bool = False):
        """ 
        General function to plot values for each molecular configuration.
        Args:
            data_key (str): Key to access data from the data dictionary.
            ax (plt.Axes): Matplotlib axis object.
            ylabel (str): Label for the y-axis.
            set_title (bool, optional): Set title for the plot. Defaults to False.
            set_xlabel (bool, optional): Set x-axis label. Defaults to False.
        """
        data = self.data[data_key]
        self.plot_data_points_with_mean_std(ax, data)

        if set_xlabel:  
            ax.set_xlabel("Configuration")
        else:
            ax.set_xticklabels([])

        if set_title:
            ax.set_title(self.dataset_name)

        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
        ax.legend(loc="upper right")
        

    def plot_data_points_with_mean_std(self, ax: plt.Axes, data: np.ndarray):
        """ 
        Plot data for each molecular configuration with mean and standard deviation lines. 
        Args:
            ax (plt.Axes): Matplotlib axis object.
            data (np.ndarray): Data to plot.
        """
        ax.plot(data, color=MAIN_COLOR, marker=".", linestyle='None', alpha=0.8, markersize=0.5)

        mean, std = data.mean(), data.std()
        ax.hlines(mean, 0, len(data), color=MEAN_COLOR, linestyle="--", linewidth=1.2, label="Mean")
        ax.hlines([mean - std, mean + std], 0, len(data), color=STD_COLOR, linestyle="--", linewidth=1.2, label=r"Mean $\pm$ Std.")

    def plot_connected_data_points(self, ax: plt.Axes, data: np.ndarray):
        """ Plot data for each molecular configuration with individual data points connected. 
        
        Args:
            ax (plt.Axes): Matplotlib axis object.
            data (np.ndarray): Data to plot.
        """
        ax.plot(data, color=MAIN_COLOR, alpha=0.8, markersize=0.5)

    def plot_values_connected(self, data_key: str, ax: plt.Axes, ylabel: str, set_title: bool = False, set_xlabel: bool = False):
        """ 
        Function to plot connected values for each molecular configuration.
        Args:
            data_key (str): Key to access data from the data dictionary.
            ax (plt.Axes): Matplotlib axis object.
            ylabel (str): Label for the y-axis.
            set_title (bool, optional): Set title for the plot. Defaults to False.
            set_xlabel (bool, optional): Set x-axis label. Defaults to False.
        """
        data = self.data[data_key]
        self.plot_connected_data_points(ax, data)
        if set_xlabel:
            ax.set_xlabel("Configuration")
        else: 
            ax.set_xticklabels([]) # Keep the ticks but hide the labels
        if set_title:
            ax.set_title(self.dataset_name)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)


    def plot_autocorrelation(self, data_key: str, ax: plt.axes, lags: int, set_title: bool = False, set_xlabel: bool = False):
        """ 
        Plot autocorrelation function for desired property (energy, force, etc.) for the dataset.
        Args:
            data_key (str): Key to access data from the data dictionary.
            ax (plt.Axes): Matplotlib axis object.
            lags (int): Number of lags to plot.
            set_title (bool, optional): Set title for the plot. Defaults to False.
            set_xlabel (bool, optional): Set x-axis label. Defaults to False.
        """
        data = self.data[data_key]
        plot_acf(data, lags=lags, use_vlines=False, title="", ax=ax, alpha=0.05)
        if set_xlabel:
            ax.set_xlabel("Lag")
        else:
            ax.set_xticklabels([])
        if set_title:
            ax.set_title(self.dataset_name)
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)  # Subtle grid lines
        # Change the dot size
        for line in ax.lines:
            line.set_markersize(1)




class MoleculeTrajectoryComparer(MoleculeTrajectoryAnalyzer):
    """ Class to produce comparison plots for one molecule across multiple datasets, e.g., MD17 and revised MD17. """
    def __init__(self, dataloaders: dict, desired_batches: int, plot_dir: str = "plots"):
        """ Initialize the TrajectoryComparer object.

        Args:
            dataloaders (dict): Dictionary containing dataloader and the names of their corresponding datasets.
            desired_batches (int): Number of batches to load from each dataset.
            plot_dir (str, optional): Directory to save plots. Defaults to "plots".
        """
        self.desired_batches = desired_batches
        self.plot_dir = plot_dir
        # Extract data for all datasets
        self.data = [self.get_data(dataset) for dataset in dataloaders.values()]
    

    def plot_distribution_comparison(self, data_key: str, axes: plt.axes, xlabel: str, set_title: bool = False, 
                                     legend_location: str = "upper right", nr_of_xticks: int = 0):
        """ 
        Plot distribution comparison for a property (energy, force, etc.) across two datasets. 
        Args:
            data_key (str): Key to access data from the data dictionary.
            axes (plt.Axes): Matplotlib axis object.
            xlabel (str): Label for the x-axis.
            set_title (bool, optional): Set title for the plot. Defaults to False.
            legend_location (str, optional): Location of the legend. Defaults to "upper right".
            nr_of_xticks (int, optional): Number of ticks on the x-axis. Defaults to 0.
        """
        for i, (ax, ds_name) in enumerate(zip(axes, self.datasets.keys())):
            data = self.data[i][data_key]
            num_bins = get_bin_number(data)
            self.plot_histogram_with_mean_std(ax, data, num_bins)
            ax.set_xlabel(xlabel)
            # Dynamically limit the number of ticks on the x-axis
            if nr_of_xticks > 0:
                ax.xaxis.set_major_locator(MaxNLocator(integer=False, prune='lower', nbins=nr_of_xticks)) 
            if set_title:
                ax.set_title(ds_name)
            ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
            ax.legend(frameon=True, loc=legend_location)
        # Y-axis label only on the first (left) subplot
        axes[0].set_ylabel("Frequency")

    def plot_values_comparison(self, data_key: str, axes: plt.Axes, ylabel: str, set_title: bool = False, set_xlabel: bool = False):
        """ Plot value comparison for a property (energy, force, etc.) across two datasets. 
        
        Args:
            data_key (str): Key to access data from the data dictionary.
            axes (plt.Axes): Matplotlib axis object.
            ylabel (str): Label for the y-axis.
            set_title (bool, optional): Set title for the plot. Defaults to False
            set_xlabel (bool, optional): Set x-axis label. Defaults
        """
        for i, (ax, ds_name) in enumerate(zip(axes, self.datasets.keys())):
            data = self.data[i][data_key]
            self.plot_data_points_with_mean_std(ax, data)
            # X-axis label
            if set_xlabel:
                ax.set_xlabel("Configuration")
            else: 
                ax.set_xticklabels([]) 
            # Title
            if set_title:
                ax.set_title(ds_name)
            ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
            ax.legend(loc="upper right")
        # Y-axis label only on the first (left) subplot
        axes[0].set_ylabel(ylabel)
        
        

    def plot_values_connected_comparison(self, data_key: str, axes: plt.Axes, ylabel: str, set_title: bool = False, set_xlabel: bool = False):
        """ Plot value comparison for a property (energy, force, etc.) across two datasets 
            with individual datapoints being connected. 
        
        Args:
            data_key (str): Key to access data from the data dictionary.
            axes (plt.Axes): Matplotlib axis object.
            ylabel (str): Label for the y-axis.
            set_title (bool, optional): Set title for the plot. Defaults to False.
            set_xlabel (bool, optional): Set x-axis label. Defaults to False.
        """
        for i, (ax, ds_name) in enumerate(zip(axes, self.datasets.keys())):
            data = self.data[i][data_key]
            self.plot_connected_data_points(ax, data)
            # X-axis label
            if set_xlabel:
                ax.set_xlabel("Configuration")
            else: 
                ax.set_xticklabels([]) # Keep the ticks but hide the labels
            # Title
            if set_title:
                ax.set_title(ds_name)
            ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
        # Y-axis label only on the first (left) subplot 
        axes[0].set_ylabel(ylabel)
        

    

    def plot_autocorrelation_comparison(self, data_key: str, axes: plt.axes, lags: int, set_title: bool = False, set_xlabel: bool = False):
        """ 
        Plot autocorrelation comparison for a property (energy, force, etc.) across two datasets. 
        Args:
            data_key (str): Key to access data from the data dictionary.
            axes (plt.Axes): Matplotlib axis object.
            lags (int): Number of lags to plot.
            set_title (bool, optional): Set title for the plot. Defaults to False.
            set_xlabel (bool, optional): Set x-axis label. Defaults to False.
        """
        for i, (ax, ds_name) in enumerate(zip(axes, self.datasets.keys())):
            data = self.data[i][data_key]
            plot_acf(data, lags=lags, use_vlines=False, title="", ax=ax, alpha=0.05)
            # X-axis label
            if set_xlabel:
                ax.set_xlabel("Lag")
            else:
                ax.set_xticklabels([])
            # Title
            if set_title:
                ax.set_title(ds_name)
            ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
            for line in ax.lines:
                line.set_markersize(1)
        # Y-axis label only on the first (left) subplot
        axes[0].set_ylabel("Autocorrelation")

