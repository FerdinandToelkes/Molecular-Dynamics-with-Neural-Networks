import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from schnetpack import properties
from schnetpack import units as spk_units
from schnetpack.md.data import HDF5Loader, PowerSpectrum
from ase.io import write

from md_with_schnet.utils import setup_logger, set_data_prefix, set_plotting_config


# Example command to run the script from within code directory:
"""
python -m md_with_schnet.md17_prediction.analyze_md_simulation
"""

def parse_args() -> dict:
    """ Parse command-line arguments. 

    Returns:
        dict: Dictionary containing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for training SchNetPack on MD17 dataset")
    parser.add_argument("-ll", "--log_level", type=str, default="INFO", help="Logging level, can be INFO, DEBUG, WARNING, ERROR, CRITICAL. Defaults to INFO.")
    return vars(parser.parse_args())

def plot_energies(data: HDF5Loader, energies_system: np.ndarray, energies_calculator: np.ndarray):
    """ Plot the energies of the system and the calculator.
    Args:
        data (HDF5Loader): HDF5Loader object containing the MD simulation data.
        energies_system (np.ndarray): Energies of the system.
        energies_calculator (np.ndarray): Energies of the calculator.
    """
    # Get the time axis in femtoseconds
    time_axis = np.arange(data.entries) * data.time_step / spk_units.fs 

    set_plotting_config()
    plt.figure()
    plt.plot(time_axis, energies_system, label=r"E$_\mathrm{pot}$ (System)")
    plt.plot(time_axis, energies_calculator, label=r"E$_\mathrm{pot}$ (Logger)", ls="--")
    plt.ylabel("E [kcal/mol]")
    plt.xlabel("t [fs]")
    plt.xlim(9800,10000)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_temperature(data: HDF5Loader):
    """ Plot the temperature of the system.
    Args:
        data (HDF5Loader): HDF5Loader object containing the MD simulation data.
    """
    set_plotting_config()

    # Read the temperature
    temperature = data.get_temperature()

    # Compute the cumulative mean
    temperature_mean = np.cumsum(temperature) / (np.arange(data.entries)+1)

    # Get the time axis
    time_axis = np.arange(data.entries) * data.time_step / spk_units.fs  # in fs

    plt.figure(figsize=(8,4))
    plt.plot(time_axis, temperature, label='T')
    plt.plot(time_axis, temperature_mean, label='T (avg.)')
    plt.ylabel('T [K]')
    plt.xlabel('t [fs]')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spectrum(frequencies: np.ndarray, intensities: np.ndarray):
    """ Plot the spectrum.
    Args:
        frequencies (np.ndarray): Frequencies of the spectrum.
        intensities (np.ndarray): Intensities of the spectrum.
    """
    set_plotting_config()
    # Plot the spectrum
    plt.figure()
    plt.plot(frequencies, intensities)
    plt.xlim(0,4000)
    plt.ylim(0,100)
    plt.ylabel('I [a.u.]')
    plt.xlabel(r'$\omega$ [cm$^{-1}$]')
    plt.show()

def main(log_level: str):
    # setup
    logger = setup_logger(log_level)
    data_prefix = set_data_prefix()
    md_workdir = f'{data_prefix}/md_workdir'
    os.makedirs(md_workdir, exist_ok=True)

    # load the HDF5 file containing the MD simulation data
    log_file = os.path.join(md_workdir, "simulation_schnet.hdf5")
    data = HDF5Loader(log_file)

    # log available properties
    log_properties = [prop for prop in data.properties]
    logger.info(f"Available properties in the HDF5 file:\n{log_properties}")

    
    # Get the energy logged via PropertiesStream
    energies_calculator = data.get_property(properties.energy, atomistic=False)
    # Get potential energies stored in the MD system
    energies_system = data.get_potential_energy()

    # Check the overall shape
    logger.debug(f"Shape: {energies_system.shape}")
    logger.debug(f"data.entries: {data.entries}")
    logger.debug(f"data.time_step (in atomic units): {data.time_step}")


    # Convert the system potential energy from internal units (kJ/mol) to kcal/mol
    energies_system *= spk_units.convert_units("kJ/mol", "kcal/mol")

    # Plot the energies
    plot_energies(data, energies_system, energies_calculator)
    logger.debug(f"Min energy: {np.min(energies_system)} kcal/mol")
    logger.debug(f"Max energy: {np.max(energies_system)} kcal/mol")
    # Plot the temperature
    plot_temperature(data)
    logger.debug(f"Min temperature: {np.min(data.get_temperature())} K")
    logger.debug(f"Max temperature: {np.max(data.get_temperature())} K")

    # extract structure information from HDF5 data
    trajectory_path = os.path.join(md_workdir, "trajectory.xyz")
    if not os.path.exists(trajectory_path):
        md_atoms = data.convert_to_atoms()

        # write list of Atoms to XYZ file
        write(
            trajectory_path,
            md_atoms,
            format="xyz"
        )

    # Initialize the spectrum
    equilibrated_data = HDF5Loader(log_file, skip_initial=10000) # i.e. skip first half
    spectrum = PowerSpectrum(equilibrated_data, resolution=2048)

    # Compute the spectrum for the first molecule (default)
    spectrum.compute_spectrum(molecule_idx=0)

    # Get frequencies and intensities
    frequencies, intensities = spectrum.get_spectrum()

    plot_spectrum(frequencies, intensities)

    



if __name__ == "__main__":
    args = parse_args()
    main(**args)