import os
import logging

from schnetpack.md.data import HDF5Loader

from md_with_schnet.utils import setup_logger, set_data_prefix, test_stuff


# Example command to run the script from within code directory:
"""
python -m md_with_schnet.md17_prediction.analyze_md_simulation
"""

def main():
    # setup
    data_prefix = set_data_prefix()
    md_workdir = f'{data_prefix}/md_workdir'
    os.makedirs(md_workdir, exist_ok=True)

    # load the HDF5 file containing the MD simulation data
    log_file = os.path.join(md_workdir, "simulation.hdf5")
    data = HDF5Loader(log_file)

    # log available properties
    log_properties = [prop for prop in data.properties]
    logger.info(f"Available properties in the HDF5 file:\n{log_properties}")


if __name__ == "__main__":
    # setup logger for this module
    logger = setup_logger(logging.DEBUG)
    
    main()