import os
import argparse
import numpy as np

from hydra import initialize, compose
from omegaconf import DictConfig


from md_with_schnet.setup_logger import setup_logger
from md_with_schnet.neural_net.inference_with_ase import update_config_with_train_config

# for interactive plotting
import plotly.graph_objects as go
# ensure that no conflict with jupyter package (ipykernel) occurs
import plotly.io as pio
pio.renderers.default = 'browser' 

logger = setup_logger("debug")


# Example command to run the script from within code directory:
"""
python -m md_with_schnet.neural_net.plot_interactive_md_ase_sim --model_dir MOTOR_MD_XTB_T300_1_ang_kcal_mol_epochs_1000_bs_100_lr_0.0001_seed_42 --simulation_name  md_sim_steps_500_time_step_0.5_seed_42 --n_samples 100
"""

def parse_args() -> dict:
    """ 
    Parse command-line arguments. 
    Returns:
        dict: Dictionary containing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for analyzing the trajectory predicted with the trained model on XTB test data.")
    # paths setup
    parser.add_argument("-mdir", "--model_dir", type=str, default="MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42", help="Directory of the trained model (default: MOTOR_MD_XTB/T300_1/epochs_1000_bs_100_lr_0.0001_seed_42)")
    parser.add_argument("--units", type=str, default="angstrom_kcal_per_mol_fs", choices=["angstrom_kcal_per_mol_fs", "angstrom_ev_fs", "bohr_hartree_aut"], help="Units for the input data (default: angstrom_kcal_per_mol_fs).")
    parser.add_argument("-sn", "--simulation_name", type=str, default="md_sim_steps_10000_time_step_0.5_seed_42", help="Name of the MD simulation (default: md_sim_steps_2000_time_step_0.5_seed_42)")
    # analysis setup
    parser.add_argument("-ns", "--n_samples", type=int, default=100, help="Number of samples to analyze (default: 1000)")
    parser.add_argument("-fs", "--first_sample", type=int, default=0, help="First sample to analyze (default: 0)")
    return vars(parser.parse_args())



def get_data_dict(xtb_data: np.ndarray, nn_data: np.ndarray) -> dict:
    """
    Extract data from XTB and NN log files.
    Args:
        xtb_data (np.ndarray): Numpy array containing XTB data.
        nn_data (np.ndarray): Numpy array containing NN data.
    Returns:
        dict: Dictionary containing time steps and energy values for both XTB and NN.
    """
    # ase saves the time in ps
    time_steps = xtb_data[:, 0] 
    if not np.array_equal(time_steps, nn_data[:, 0]):
        raise ValueError("Time steps in XTB and NN data do not match.")
    
    xtb_e_tot = xtb_data[:, 1]  
    xtb_e_pot = xtb_data[:, 2]  
    xtb_e_kin = xtb_data[:, 3]  
    xtb_temp = xtb_data[:, 4]  
    
    nn_e_tot = nn_data[:, 1]  
    nn_e_pot = nn_data[:, 2]  
    nn_e_kin = nn_data[:, 3]  
    nn_temp = nn_data[:, 4]  

    log_data = {
        "XTB": {
            "time_steps": time_steps,
            "e_tot": xtb_e_tot,
            "e_pot": xtb_e_pot,
            "e_kin": xtb_e_kin,
            "temp": xtb_temp
        },
        "NN": {
            "time_steps": time_steps,
            "e_tot": nn_e_tot,
            "e_pot": nn_e_pot,
            "e_kin": nn_e_kin,
            "temp": nn_temp
        }
    }
    return log_data

def prepare_properties_data(data: dict) -> tuple:
    """
    Prepare properties data for interactive plotting.
    Args:
        data (dict): Dictionary containing XTB and NN properties with keys:
                     "time_steps", "e_tot", "e_pot", "e_kin", "temp".
    Returns:
        tuple: A tuple containing a dictionary of properties and a dictionary of y-axis labels.
    """ 
    # scale total energies to [0, 1] range for better visualization
    xtb_e_tot_scaled = (data["XTB"]["e_tot"] - np.min(data["XTB"]["e_tot"])) / (np.max(data["XTB"]["e_tot"]) - np.min(data["XTB"]["e_tot"]))
    nn_e_tot_scaled = (data["NN"]["e_tot"] - np.min(data["NN"]["e_tot"])) / (np.max(data["NN"]["e_tot"]) - np.min(data["NN"]["e_tot"]))

    # Define properties
    properties = {
        "Total Energy": {"xtb": data["XTB"]["e_tot"], "nn": data["NN"]["e_tot"]},
        "Scaled Total Energy": {"xtb": xtb_e_tot_scaled, "nn": nn_e_tot_scaled},
        "Potential Energy": {"xtb": data["XTB"]["e_pot"], "nn": data["NN"]["e_pot"]},
        "Kinetic Energy": {"xtb": data["XTB"]["e_kin"], "nn": data["NN"]["e_kin"]},
        "Temperature": {"xtb": data["XTB"]["temp"], "nn": data["NN"]["temp"]},
    }

    # A small lookup table for y-axis labels:
    y_labels = {
        "Total Energy": "Energy [eV]",
        "Scaled Total Energy": "Normalized Value",
        "Potential Energy": "Energy [eV]",
        "Kinetic Energy": "Energy [eV]",
        "Temperature": "Temperature [K]",
    }
    return properties, y_labels

def create_interactive_properties_plot(properties: dict, time_steps: np.ndarray, y_labels: dict, plot_dir: str):
    """
    Create an interactive plot with dropdown buttons to select different properties.
    Args:
        properties (dict): Dictionary containing the properties to plot.
        time_steps (np.ndarray): Array of time steps.
        y_labels (dict): Dictionary mapping property names to y-axis labels.
        plot_dir (str): Directory to save the plot.
    """
    #### Build the figure ##################################################
    fig = go.Figure()

    all_props = list(properties.keys())
    for i, prop_name in enumerate(all_props):
        y_xtb = properties[prop_name]["xtb"]
        y_nn  = properties[prop_name]["nn"]
        
        # Add the XTB trace:
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=y_xtb,
                mode="lines",
                name=f"XTB {prop_name}",
                visible=(i == 0),  # only the first property is visible at start
            )
        )
        # Add the NN trace:
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=y_nn,
                mode="lines",
                name=f"NN {prop_name}",
                visible=(i == 0),
            )
        )

    #### Create dropdown buttons ###########################################
    buttons = []
    for i, prop_name in enumerate(all_props):
        # Build a visibility mask: 2 traces per property, so length = 2 * num_properties.
        visibility = [False] * (len(all_props) * 2)
        visibility[2*i]     = True  # XTB trace for this property
        visibility[2*i + 1] = True  # NN trace for this property

        buttons.append(
            dict(
                label=prop_name,
                method="update",
                args=[
                    {"visible": visibility},     # which traces to show
                    {
                        "title": {
                            "text": f"{prop_name} over Time",
                            "x": 0.5,
                            "xanchor": "center",
                        },
                        "yaxis": {
                            "title": {"text": y_labels[prop_name]}
                        }
                    },
                ],
            )
        )

    #### Initial layout ##################################################
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
            )
        ],
        title={
            "text": f"{all_props[0]} over Time",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Time [ps]",
        yaxis_title=y_labels[all_props[0]],  # e.g. “Energy [eV]” for “Potential Energy”
        margin={"t": 100},  # leave enough room at the top for dropdown + title
    )
    # save the figure to a file
    fig.write_html(f"{plot_dir}/interactive_properties_plot.html")

    fig.show()

def create_interactive_window_scatter(xtb_prop: np.ndarray, nn_prop: np.ndarray, window_sizes: list, plot_dir: str):
    """
    Create an interactive Plotly scatter where:
      - A dropdown lets you pick one window size W from window_sizes.
      - A slider then controls the “first_config” index i, showing the scatter of
        xtb_prop[i : i+W] vs. nn_prop[i : i+W].

    Args:
        xtb_prop (np.ndarray): XTB property values (e.g., potential energy).
        nn_prop (np.ndarray): NN property values (e.g., potential energy).
        window_sizes (list of int): List of candidate window-sizes to choose from.
        plot_dir (str): Directory to save the plot.
    """
    # Make sure the inputs are the same length:
    n_points = len(xtb_prop)

    # Precompute, for each window_size W, the slider steps array
    # Each step: method="update", args[0] changes the trace’s "x" and "y" slices.
    sliders_for_W = {}
    for W in window_sizes:
        max_start = n_points - W  # last valid starting index is n_points - W
        steps = []
        for i in range(max_start + 1):
            start = i
            end = i + W
            slice_x = xtb_prop[start:end]
            slice_y = nn_prop[start:end]

            step = dict(
                method="update",
                args=[
                    # args[0]: update data of the single trace:
                    {
                        "x": [slice_x],
                        "y": [slice_y]
                    },
                    # args[1]: we update the title to show current window and indices:
                    {
                        "title": {
                            "text": f"Scatter (W={W}, i={start}…)​",
                            "x": 0.5,
                            "xanchor": "center"
                        }
                    }
                ],
                label=str(i)
            )
            steps.append(step)

        sliders_for_W[W] = [
            dict(
                active=0,
                currentvalue={"prefix": "start index: "},
                pad={"t": 50},
                steps=steps
            )
        ]

    # Create the figure and add one initial Scatter trace (using the first window_size)
    initial_W = window_sizes[0]
    init_slice_x = xtb_prop[0 : initial_W]
    init_slice_y = nn_prop[0 : initial_W]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=init_slice_x,
                y=init_slice_y,
                mode="markers",
                marker={"size": 3, "color": "black", "opacity": 1.0},
                name=f"W={initial_W}"
            )
        ]
    )

    # Build dropdown buttons: each button swaps in the appropriate initial slice + slider
    buttons = []
    for W in window_sizes:
        # The “initial” slice for this W is i=0…W
        init_x = xtb_prop[0:W]
        init_y = nn_prop[0:W]

        buttons.append(
            dict(
                label=f"Window Size={W}",
                method="update",
                args=[
                    # args[0]: update the trace’s x/y to the new window’s i=0 slice
                    {
                        "x": [init_x],
                        "y": [init_y],
                        "marker": {"size": 3, "color": "black", "opacity": 1.0}
                    },
                    # args[1]: update layout → title & sliders
                    {
                        "title": {
                            "text": f"Scatter (W={W}, i=0…)",
                            "x": 0.5,
                            "xanchor": "center"
                        },
                        "sliders": sliders_for_W[W]
                    }
                ]
            )
        )

    # Attach dropdown and the initial slider to layout
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True
            )
        ],
        sliders=sliders_for_W[initial_W],  # attach the initial window_size’s slider
        title={
            "text": f"Scatter (W={initial_W}, i=0…)",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title="XTB Potential Energy [eV]",
        yaxis_title="NN  Potential Energy [eV]",
        margin={"t": 100}  # extra top margin for dropdown + title
    )
    # Save the figure to a file
    # too large to be displayed without additional configuration
    fig.write_html(f"{plot_dir}/interactive_window_scatter.html")

    fig.show()


def rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the rolling Pearson correlation between two 1D arrays.

    Parameters:
    - a (np.ndarray): First input array.
    - b (np.ndarray): Second input array.
    - window (int): Size of the rolling window.

    Returns:
    - np.ndarray: Array of rolling correlation coefficients.
    """
    if len(a) != len(b):
        raise ValueError("Input arrays must have the same length.")
    if window > len(a):
        raise ValueError("Window size must be less than or equal to the length of the input arrays.")

    # Number of rolling windows
    n = len(a) - window + 1

    # Initialize array to store correlation coefficients
    corr = np.empty(n)

    for i in range(n):
        a_window = a[i:i+window]
        b_window = b[i:i+window]

        # Compute means
        a_mean = a_window.mean()
        b_mean = b_window.mean()

        # Compute numerator and denominator for Pearson correlation
        numerator = np.sum((a_window - a_mean) * (b_window - b_mean))
        denominator = np.sqrt(np.sum((a_window - a_mean)**2) * np.sum((b_window - b_mean)**2))

        # Handle division by zero
        if denominator == 0:
            corr[i] = np.nan
        else:
            corr[i] = numerator / denominator

    return corr


def compute_rolling_correlations(data: dict, window_sizes: list) -> dict:
    """
    Compute rolling correlations between XTB and NN potential and kinetic energies for given window sizes.
    Args:
        data (dict): Dictionary containing XTB and NN properties with keys:
        window_sizes (list of int): List of rolling‐window sizes.
    Returns:
        dict: Dictionary containing rolling correlation data for potential and kinetic energies.
    """    
    # Store, for each property, the list (time_array, corr_array) tuples, one entry per window_size
    rolling_data = {
        "Potential Energy": [],
        "Kinetic Energy": []
    }
    
    original_time_steps = data["XTB"]["time_steps"]
    for w in window_sizes:
        corr_pot = rolling_corr(data["XTB"]["e_pot"], data["NN"]["e_pot"], w)
        corr_kin = rolling_corr(data["XTB"]["e_kin"], data["NN"]["e_kin"], w)

        # new time steps = time steps - window_size + 1
        time_corr = original_time_steps[:-(w-1)]

        rolling_data["Potential Energy"].append((time_corr, corr_pot))
        rolling_data["Kinetic Energy"].append((time_corr, corr_kin))
    return rolling_data


def create_interactive_rolling_corr_plot(rolling_data: dict, window_sizes: list, plot_dir: str):
    """
    Create an interactive Plotly figure showing rolling correlations
    between XTB and NN for two properties (Potential and Kinetic energy)
    across multiple window sizes. A dropdown menu allows switching
    between properties, and each property’s subplot contains one trace
    per window size.
    
    Args:
        rolling_data (dict): Dictionary containing rolling correlation data.
        window_sizes (list of int): List of rolling‐window sizes.
        plot_dir (str): Directory to save the plot.
    """
    
    
    # Now build the Plotly figure
    fig = go.Figure()
    all_props = ["Potential Energy", "Kinetic Energy"]
    
    # Add one trace per (property, window_size). We'll keep them all in the same figure,
    # but only make one property’s traces visible at a time.
    for i, prop_name in enumerate(all_props):
        for j, w in enumerate(window_sizes):
            t_arr, corr_arr = rolling_data[prop_name][j]
            fig.add_trace(
                go.Scatter(
                    x=t_arr,
                    y=corr_arr,
                    mode="lines",
                    name=f"Window {w}",
                    visible=(i == 0),  # Only show the first property’s traces initially
                    legendgroup=str(w),  # group legends by window size if desired
                )
            )
    
    # Create two dropdown buttons (one for each property)
    buttons = []
    num_windows = len(window_sizes)
    total_traces = len(all_props) * num_windows  # = 2 * len(window_sizes)
    
    for i, prop_name in enumerate(all_props):
        # Build a mask of length total_traces, turning on the i-th block of size num_windows
        visibility = [False] * total_traces
        start_index = i * num_windows
        for k in range(num_windows):
            visibility[start_index + k] = True
        
        buttons.append(
            dict(
                label=prop_name,
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": {
                            "text": f"Rolling Correlation: {prop_name}",
                            "x": 0.5,
                            "xanchor": "center"
                        },
                        "yaxis": {
                            "title": {"text": "Rolling Correlation"}
                        }
                    }
                ]
            )
        )
    
    # Final layout tweaks
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True
            )
        ],
        title={
            "text": f"Rolling Correlation: {all_props[0]}",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title="Time [ps]",
        yaxis_title="Rolling Correlation",
        margin={"t": 100}
    )
    # Save the figure to a file
    fig.write_html(f"{plot_dir}/interactive_rolling_corr_plot.html")

    fig.show()


def main(model_dir: str, units: str, simulation_name: str, n_samples: int, first_sample: int):
    ####################### 1) Compose the config ###########################
    with initialize(config_path=f"conf/{units}", job_name="inference", version_base="1.1"):
        cfg: DictConfig = compose(config_name="inference_config")

    # use training config to update the inference config
    train_cfg_path = os.path.join("runs", units, model_dir, "tensorboard/default/version_0")
    with initialize(config_path=train_cfg_path, job_name="train", version_base="1.1"):
        cfg_train: DictConfig = compose(config_name="hparams.yaml")
    cfg = update_config_with_train_config(cfg, cfg_train)

    ####################### 2) Prepare Data and Paths #########################
    home_dir = os.path.expanduser("~")
    runs_dir_path = os.path.join(home_dir, cfg.globals.runs_dir_subpath)
    model_dir_path = os.path.join(runs_dir_path, units, model_dir)
    logger.debug(f"model_dir_path: {model_dir_path}")
    logger.debug(f"simulation_name: {simulation_name}")
    nn_target_dir = os.path.join(model_dir_path, simulation_name)
    xtb_target_dir = os.path.join(runs_dir_path, cfg.globals.xtb_dir_name, simulation_name)
    logger.debug(f"nn_target_dir: {nn_target_dir}")
    logger.debug(f"xtb_target_dir: {xtb_target_dir}")


    # load the log file containing the MD simulation data
    xtb_data = np.loadtxt(f'{xtb_target_dir}/xtb_md.log', skiprows=1) 
    nn_data = np.loadtxt(f'{nn_target_dir}/nn_md.log', skiprows=1) # time[ps], Etot, Epot, Ekin[eV], T[K]
    # nn_data = np.loadtxt(f'{nn_target_dir}/xtb_md.log', skiprows=1) # time[ps], Etot, Epot, Ekin[eV], T[K]
    logger.debug(f"Shape of xtb_data: {xtb_data.shape}")
    logger.debug(f"Shape of nn_data: {nn_data.shape}")

    # take first n entries for plotting
    if n_samples + first_sample > xtb_data.shape[0]:
        raise ValueError(f"Requested n_samples ({n_samples}) + first_sample ({first_sample}) exceeds available data length ({xtb_data.shape[0]}).")
    xtb_data = xtb_data[first_sample:first_sample + n_samples, :]  # time[ps], Etot, Epot, Ekin[eV], T[K]
    nn_data = nn_data[first_sample:first_sample + n_samples, :]  # time[ps], Etot, Epot, Ekin[eV], T[K]

    log_data = get_data_dict(xtb_data, nn_data)
    logger.debug(f"Log data keys: {log_data.keys()}")

    
    ####################### 3) Make interactive Plots #########################
    properties, y_labels = prepare_properties_data(log_data)
    plot_dir = os.path.join("md_with_schnet/neural_net/plots", model_dir, simulation_name)
    # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    logger.debug(f"Plot directory: {plot_dir}")

    # Create interactive plot with dropdown
    create_interactive_properties_plot(
        properties=properties,
        time_steps=log_data["XTB"]["time_steps"],
        y_labels=y_labels,
        plot_dir=plot_dir
    )

    window_sizes = [100, 250, 500, 1000]
    
    # Create an interactive scatter plot with a dropdown for window sizes
    create_interactive_window_scatter(
        xtb_prop=log_data["XTB"]["e_pot"],
        nn_prop=log_data["NN"]["e_pot"],
        window_sizes=window_sizes,
        plot_dir=plot_dir
    )

    # Compute rolling correlations for potential and kinetic energies
    rolling_data = compute_rolling_correlations(data=log_data, window_sizes=window_sizes)

    create_interactive_rolling_corr_plot(
        rolling_data=rolling_data,
        window_sizes=window_sizes,
        plot_dir=plot_dir
    )



    

if __name__ == "__main__":
    args = parse_args()
    main(**args)


# code for non interactive plotting
# # plot potential energies from xtb and nn
# set_plotting_config(fontsize=10, aspect_ratio=468/525, width_fraction=1)
# fig, axes = plt.subplots(3) 
# axes[0].plot(time_steps, xtb_e_tot, label='XTB Total Energy', color='green')
# axes[0].plot(time_steps, nn_e_tot, label='NN Total Energy', color='red')
# axes[0].set_xlabel('Time [ps]')
# axes[0].set_ylabel('Total Energy [eV]')
# axes[0].set_title('Total Energy Comparison')
# axes[0].legend()
# axes[1].plot(time_steps, xtb_e_pot, label='XTB Potential Energy', color='cyan')
# axes[1].plot(time_steps, nn_e_pot, label='NN Potential Energy', color='magenta')
# axes[1].set_xlabel('Time [ps]')
# axes[1].set_ylabel('Potential Energy [eV]')
# axes[1].set_title('Potential Energy Comparison')
# axes[1].legend()
# axes[2].plot(time_steps, xtb_e_kin, label='XTB Kinetic Energy', color='purple')
# axes[2].plot(time_steps, nn_e_kin, label='NN Kinetic Energy', color='brown')
# axes[2].set_xlabel('Time [ps]')
# axes[2].set_ylabel('Kinetic Energy [eV]')
# axes[2].set_title('Kinetic Energy Comparison')
# axes[2].legend()
# plt.tight_layout()
# plt.show()

# # plot potential energies on twin axes
# set_plotting_config(fontsize=10, aspect_ratio=8/4, width_fraction=1)
# fig, ax1 = plt.subplots()
# ax1.plot(time_steps, xtb_e_pot, label='XTB Potential Energy', color='green')
# ax2 = ax1.twinx()  # create a twin Axes sharing the x-axis
# ax2.plot(time_steps, nn_e_pot, label='NN Potential Energy', color='red')
# ax1.set_xlabel('Time [ps]')
# ax1.set_ylabel('XTB Potential Energy [eV]', color='green')
# ax2.set_ylabel('NN Potential Energy [eV]', color='red')
# ax1.set_title('Potential Energy Comparison')
# ax1.tick_params(axis='y', labelcolor='green')
# ax2.tick_params(axis='y', labelcolor='red')
# # Combine legends from both axes
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines + lines2, labels + labels2, loc='upper left')
# plt.tight_layout()
# plt.show()

# # compute rolling correlation with pandas
# xtb_df = pd.DataFrame({
#     'time': time_steps,
#     'e_pot': xtb_e_pot,
#     'e_kin': xtb_e_kin,
#     'temp': xtb_temp
# })
# nn_df = pd.DataFrame({
#     'time': time_steps,
#     'e_pot': nn_e_pot,
#     'e_kin': nn_e_kin,
#     'temp': nn_temp
# })
# # compute rolling correlation with a window of 100
# window_sizes = [100, 250, 500]
# # take first four colors from the default matplotlib color cycle
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(window_sizes)]

# xtb_nn_corrs = []
# xtb_nn_time = []
# for window_size in window_sizes:
#     xtb_nn_corr = xtb_df.rolling(window=window_size).corr(nn_df)
#     xtb_nn_corr = xtb_nn_corr.dropna()
#     xtb_nn_corrs.append(xtb_nn_corr)
#     # new time steps = time steps - window_size + 1
#     xtb_nn_time.append(xtb_df['time'][:-(window_size-1)]) 
#     logger.debug(f"XTB correlation shape: {xtb_nn_corr.shape}")

# # plot the rolling correlation
# set_plotting_config(fontsize=10, aspect_ratio=8/4, width_fraction=1)
# plt.figure()
# for i, window_size in enumerate(window_sizes):
#     xtb_nn_corr = xtb_nn_corrs[i]
#     xtb_nn_time_i = xtb_nn_time[i]
#     xtb_nn_time_i = xtb_nn_time[i]
#     plt.plot(xtb_nn_time_i, xtb_nn_corr['e_pot'], label=f'Window Size {window_size}', color=colors[i])
# plt.xlabel('Time [ps]')
# plt.ylabel('Rolling Correlation')
# plt.title('Rolling Correlation between XTB and NN Potential Energies')
# plt.legend()
# plt.tight_layout()
# plt.show()
