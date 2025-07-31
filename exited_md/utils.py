import os

from hydra import initialize, compose
from omegaconf import DictConfig

from md_with_schnet.setup_logger import setup_logger

logger = setup_logger("debug")



def get_split_path(data_prefix: str, fold: int = 0) -> str:
    """
    Get the path to the split file for the given trajectory directory and fold.
    Args:
        data_prefix (str): The prefix path to the data directory.
        trajectory_dir (str): The directory containing the trajectory data.
        fold (int): The fold number for cross-validation (default: 0).
    Returns:
        str: The path to the split file.
    """
    split_file = os.path.join(data_prefix, "splits", f"inner_splits_{fold}.npz")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Missing split file: {split_file}")
    logger.debug(f"Split file: {split_file}")
    return split_file


def remove_splitting_lock_file() -> None:
    """
    Remove the splitting.lock file if it exists in the current working directory.
    This is used to ensure that the script can run without being blocked by a previous run.
    """
    logger.info(f"Removing splitting.lock file if it exists in the current working directory ({os.getcwd()}).")
    lock_file = "./splitting.lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)

# had to be copied because it works with relative paths
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