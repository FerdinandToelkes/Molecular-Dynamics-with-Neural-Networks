import os
import argparse
import logging
import torch
import schnetpack as spk

from ase import Atoms
from schnetpack import properties
from schnetpack.md import System, UniformInit, Simulator
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import ASENeighborList
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack.md.simulation_hooks import LangevinThermostat, callback_hooks


from md_with_schnet.utils import setup_logger, load_md17_dataset, set_data_prefix


# Example command to run the script from within code directory:
"""
python -m md_with_schnet.md17_prediction.inference
"""

logger = setup_logger(logging.INFO)

def parse_args() -> dict:
    """ Parse command-line arguments. 

    Returns:
        dict: Dictionary containing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for training SchNetPack on MD17 dataset")
    parser.add_argument("--dataset_name", type=str, default="rMD17", help="Name of the dataset to load from (default: rMD17)")
    parser.add_argument("--molecule_name", type=str, default="ethanol", help="Name of the molecule to load (default: ethanol)")
    return vars(parser.parse_args())



def main(dataset_name: str, molecule_name: str):
    # setup
    data_prefix = set_data_prefix()
    output_dir = f'{data_prefix}/output'
    md_workdir = f'{data_prefix}/md_workdir'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(md_workdir, exist_ok=True)

    # set device and load model
    device = torch.device("cuda")
    model_path = os.path.join(output_dir, "best_inference_model")
    best_model = torch.load(model_path, map_location=device)

    # load original and revised MD17 dataset (e.g., Ethanol)
    data = load_md17_dataset(data_prefix, molecule=molecule_name, dataset_name=dataset_name)
    logger.info(f"loaded dataset: {data}")

    # pick starting structure from test dataset
    test_dataset = data.test_dataset
    structure = test_dataset[0]  

    # load molecule with ASE
    molecule = Atoms(
        numbers=structure[spk.properties.Z],
        positions=structure[spk.properties.R]
    )

    # Number of molecular replicas
    n_replicas = 1

    # set up MD system
    md_system = System()
    md_system.load_molecules(
        molecule, 
        n_replicas=n_replicas, 
        position_unit_input="Angstrom"
        )
    
    # choose starting velocities
    system_temperature = 300 # Kelvin

    # Set up the initializer
    md_initializer = UniformInit(
        system_temperature,
        remove_center_of_mass=True,
        remove_translation=True,
        remove_rotation=True,
    )

    # Initialize the system momenta
    md_initializer.initialize_system(md_system)

    time_step = 0.5 # fs

    # Set up the integrator
    md_integrator = VelocityVerlet(time_step)
        
    # set cutoff and buffer region
    cutoff = 5.0  # Angstrom (units used in model)
    cutoff_shell = 2.0  # Angstrom

    # initialize neighbor list for MD using the ASENeighborlist as basis
    md_neighborlist = NeighborListMD(
        cutoff,
        cutoff_shell,
        ASENeighborList,
    )

    md_calculator = SchNetPackCalculator(
        model_path,  # path to stored model
        "forces",  # force key
        "kcal/mol",  # energy units
        "Angstrom",  # length units
        md_neighborlist,  # neighbor list
        energy_key="energy",  # name of potential energies
        required_properties=[],  # additional properties extracted from the model
    )


    # check if a GPU is available and use a CPU otherwise
    if torch.cuda.is_available():
        md_device = "cuda"
    else:
        md_device = "cpu"

    # use single precision
    md_precision = torch.float32

    # Set thermostat constant
    time_constant = 100  # fs

    # Initialize the thermostat
    langevin = LangevinThermostat(system_temperature, time_constant)

    simulation_hooks = [
        langevin
    ]

    # Path to database
    log_file = os.path.join(md_workdir, "simulation.hdf5")

    # Size of the buffer
    buffer_size = 100

    # Set up data streams to store positions, momenta and the energy
    data_streams = [
        callback_hooks.MoleculeStream(store_velocities=True),
        callback_hooks.PropertyStream(target_properties=[properties.energy]),
    ]

    # Create the file logger
    file_logger = callback_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams,
        every_n_steps=1,  # logging frequency
        precision=32,  # floating point precision used in hdf5 database
    )

    # Update the simulation hooks
    simulation_hooks.append(file_logger)

    #Set the path to the checkpoint file
    chk_file = os.path.join(md_workdir, 'simulation.chk')

    # Create the checkpoint logger
    checkpoint = callback_hooks.Checkpoint(chk_file, every_n_steps=100)

    # Update the simulation hooks
    simulation_hooks.append(checkpoint)

    # directory where tensorboard log will be stored to
    tensorboard_dir = os.path.join(md_workdir, 'logs')

    tensorboard_logger = callback_hooks.TensorBoardLogger(
        tensorboard_dir,
        ["energy", "temperature"], # properties to log
    )

    # update simulation hooks
    simulation_hooks.append(tensorboard_logger)

    md_simulator = Simulator(md_system, md_integrator, md_calculator, simulator_hooks=simulation_hooks)

    md_simulator = md_simulator.to(md_precision)
    md_simulator = md_simulator.to(md_device)

    n_steps = 20000

    md_simulator.simulate(n_steps)

if __name__ == "__main__":
    args = parse_args()
    main(**args)