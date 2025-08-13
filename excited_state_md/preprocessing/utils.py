import os


from ground_state_md.setup_logger import setup_logger

logger = setup_logger("info")



##############################################################################################################################

def prepare_last_excited_cycles(data_path: str, computed_cycles: int) -> dict:
    """
    Prepare the last excited cycles by reading them from the file, deleting invalid cycles, and excluding directories.
    Args:
        data_path (str): Path to the directory containing the last excited cycles file.
        computed_cycles (int): The maximal number of cycles for which the ground state gradients were computed.
                               This may not be needed in the future.
    Returns:
        dict: Dictionary with keys as GEO directories and values as the last excited cycle number.
    """
    # read in the last_excited_cycle_of_valid_trajectories.txt and extract the last excited cycles
    last_excited_cycles = read_last_excited_cycles(data_path)

    # delete all entries corresponding to cycles larger than the computed cycles
    last_excited_cycles = delete_invalid_cycles(last_excited_cycles, computed_cycles)

    # read in the "excluded_directories" file (format is GEO_1\n, GEO_2\n, ...)
    excluded_directories = read_excluded_directories(data_path)

    # delete all entries in last_excited_cycles that are in excluded_directories
    last_excited_cycles = delete_excluded_directories(last_excited_cycles, excluded_directories)

    # sort them by name
    last_excited_cycles = dict(sorted(last_excited_cycles.items()))
    return last_excited_cycles

def read_last_excited_cycles(data_path: str) -> dict:
    """
    Read the last excited cycles from the file and return them as a dictionary.
    Args:
        data_path (str): Path to the directory containing the last excited cycles file.
    Returns:
        dict: Dictionary with keys as geometry directories and values as the last excited cycle number.
    """
    last_excited_cycles = {}
    with open(os.path.join(data_path, "last_excited_cycle_of_valid_trajectories.txt"), 'r') as f:
        for line in f:
            geo_dir, last_cycle = line.strip().split(": ")
            last_excited_cycles[geo_dir] = int(last_cycle)
    logger.info(f"Number of generally valid trajectories: {len(last_excited_cycles)}")
    return last_excited_cycles

def delete_invalid_cycles(last_excited_cycles: dict, computed_cycles: int) -> dict:
    """
    Delete entries in last_excited_cycles that have a last excited cycle larger than the computed cycles.
    Args:
        last_excited_cycles (dict): Dictionary with keys as geometry directories and values as the last excited cycle number.
        computed_cycles (int): The number of cycles for which the gradients were computed.
    """
    for geo_dir in list(last_excited_cycles.keys()):
        if last_excited_cycles[geo_dir] > computed_cycles:
            logger.debug(f"Deleting {geo_dir} with last excited cycle {last_excited_cycles[geo_dir]} larger than {computed_cycles}")
            del last_excited_cycles[geo_dir]
    logger.info(f"Number of trajectories after deleting cycles larger than {computed_cycles}: {len(last_excited_cycles)}")
    return last_excited_cycles

def read_excluded_directories(data_path: str) -> list:
    """
    Read the missing files from the file and return them as a list.
    Args:
        data_path (str): Path to the directory containing the missing files file.
    Returns:
        list: List of geometry directories that are missing.
    """
    excluded_directories = []
    with open(os.path.join(data_path, "from_preprocessing_excluded_directories.txt"), 'r') as f:
        for line in f:
            if line.strip() != "" and not line.startswith("#"):               
                # add the line to excluded_directories if it is not empty
               excluded_directories.append(line.strip())
    logger.debug(f"excluded_directories: {excluded_directories}")
    return excluded_directories

def delete_excluded_directories(last_excited_cycles: dict, excluded_directories: list) -> dict:
    """
    Delete entries in last_excited_cycles that are in the excluded_directories list.
    Args:
        last_excited_cycles (dict): Dictionary with keys as geometry directories and values as the last excited cycle number.
        excluded_directories (list): List of geometry directories that were excluded from ground state calculations.
    Returns:
        dict: Updated dictionary with entries removed that are in the excluded_directories list.
    """
    for geo_dir in excluded_directories:
        if geo_dir in last_excited_cycles:
            logger.debug(f"Deleting {geo_dir} from last_excited_cycles because it is in excluded_directories")
            del last_excited_cycles[geo_dir]
    logger.info(f"Number of trajectories after deleting missing files and invalid cycles: {len(last_excited_cycles)}")
    return last_excited_cycles

##############################################################################################################################

def set_path_and_remove_old_file(data_path: str, geo_dir: str, name: str) -> str:
    """
    Set the output path and remove any old files at that path if they exist.
    Args:
        data_path (str): Path to the directory containing all the data, i.e. all the GEO directories.
        geo_dir (str): Name of directory for which the energies are being processed.
        name (str): Name of the output file.
    Returns:
        str: The output path for the energies file.
    """
    output_path = os.path.join(data_path, geo_dir, "test", name)
    if os.path.exists(output_path):
        logger.info(f"Removing old file {output_path}")
        os.remove(output_path)
    return output_path

##############################################################################################################################