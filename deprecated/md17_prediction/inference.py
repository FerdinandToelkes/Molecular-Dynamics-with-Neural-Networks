import os
import argparse
import torch

from ase.io import read
from schnetpack import properties
from schnetpack.md import System, UniformInit, Simulator
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import ASENeighborList
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack.md.simulation_hooks import LangevinThermostat, callback_hooks

from schnetpack.utils import load_model


from ground_state_md.utils import set_data_prefix
from ground_state_md.setup_logger import setup_logger


# Example command to run the script from within code directory:
"""
python -m md_with_schnet.md17_prediction.inference
"""


def parse_args() -> dict:
    """ Parse command-line arguments. 

    Returns:
        dict: Dictionary containing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for training SchNetPack on MD17 dataset")
    parser.add_argument("--dataset_name", type=str, default="rMD17", help="Name of the dataset to load from (default: rMD17)")
    parser.add_argument("--molecule_name", type=str, default="ethanol", help="Name of the molecule to load (default: ethanol)")
    return vars(parser.parse_args())

logger = setup_logger("debug")

def main(dataset_name: str, molecule_name: str):
    # setup
    data_prefix = set_data_prefix()
    output_dir = f'{data_prefix}/output'
    md_workdir = f'{data_prefix}/md_workdir'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(md_workdir, exist_ok=True)


    # # set device and load model
    # model_path = os.path.join(output_dir, "best_inference_model")

    # # load original and revised MD17 dataset (e.g., Ethanol)
    # data = load_md17_dataset(data_prefix, molecule=molecule_name, dataset_name=dataset_name)
    # logger.info(f"loaded dataset: {data}")

    # # pick starting structure from test dataset
    # test_dataset = data.test_dataset
    # structure = test_dataset[0]  

    # # load molecule with ASE
    # molecule = Atoms(
    #     numbers=structure[spk.properties.Z],
    #     positions=structure[spk.properties.R]
    # )

    
    # Load model and structure (downloaded from Github)
    test_path = f"{data_prefix}/schnetpack_test"
    model_path = os.path.join(test_path, 'md_ethanol.model')
    molecule_path = os.path.join(test_path, 'md_ethanol.xyz')
    # Load atoms with ASE
    molecule = read(molecule_path)

    # investigate model structure
    model = load_model(model_path)
    logger.info(f"Model: {model}")
    


    ### SETUP THE MOLECULAR DYNAMICS SIMULATION ###
    # Number of molecular replicas
    n_replicas = 1
    system_temperature = 300 # Kelvin
    time_step = 0.5 # fs
    # set cutoff and buffer region for neighbor list
    cutoff = 5.0  # Angstrom (units used in model)
    cutoff_shell = 2.0  # Angstrom
    md_precision = torch.float32
    # Set thermostat constant
    time_constant = 100  # fs
    # Size of the buffer for the file logger
    buffer_size = 100
    # Number of steps to simulate
    n_steps = 20000

    # set up MD system
    md_system = System()
    md_system.load_molecules(
        molecule, 
        n_replicas=n_replicas, 
        position_unit_input="Angstrom"
        )
    
    # Set up the initializer
    md_initializer = UniformInit(
        system_temperature,
        remove_center_of_mass=True,
        remove_translation=True,
        remove_rotation=True,
    )

    # Initialize the system momenta
    md_initializer.initialize_system(md_system)

    # Set up the integrator
    md_integrator = VelocityVerlet(time_step)
        
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

    # Initialize the thermostat and set it as a simulation hook
    langevin = LangevinThermostat(system_temperature, time_constant)
    simulation_hooks = [
        langevin
    ]

    # Path to database
    log_file = os.path.join(md_workdir, "simulation_schnet.hdf5")

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

    # Set the path to the checkpoint file and create checkpoint logger
    chk_file = os.path.join(md_workdir, 'simulation.chk')
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

    # Set up the MD simulator
    md_simulator = Simulator(md_system, md_integrator, md_calculator, simulator_hooks=simulation_hooks)
    md_simulator = md_simulator.to(md_precision)
    md_simulator = md_simulator.to(md_device)

    # Simulate the MD system
    md_simulator.simulate(n_steps)

if __name__ == "__main__":
    args = parse_args()
    main(**args)