import os
import logging
import argparse

from md_with_schnet.utils import set_data_prefix
from md_with_schnet.setup_logger import setup_logger

# Example command to run the script from within code directory:
"""
python -m md_with_schnet.data_analysis.log2grads
"""

logger = setup_logger(logging_level_str="debug")
# # Create a logger
# level = "DEBUG"
# external_level = "WARNING"
# logger = logging.getLogger(__name__)
# logger.setLevel(level)

# # Don't propagate to root logger to avoid duplicate messages
# # which are I think caused by the schnetpack logger (bad practice)
# logger.propagate = False

# # Create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # Create console handler and set level to debug
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(level)
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)

# # Reduce noise from schnetpack and possibly other libraries
# for external_module in ["schnetpack"]:
#     logging.getLogger(external_module).setLevel(external_level)


if __name__ == "__main__":
    # setup
    data_path = set_data_prefix() + "/turbo_test"
    command_path = os.path.expanduser('~/whk/code/md_with_schnet/data_analysis/extract_gradients.sh')
    output_path = os.path.join(data_path, "gradients.txt")
    if os.path.exists(output_path):
        logger.debug("Removing old gradients.txt file")
        os.remove(output_path)
    
    # log some info
    logger.debug(f"data_path: {data_path}")
    logger.debug(f"command_path: {command_path}")
    logger.debug(f"output_path: {output_path}")
    
    # list all files in data_path
    log_files = os.listdir(data_path)
    log_files = [f for f in log_files if f.startswith("mdlog.")]

    # sort log files by their number (after the dot)
    log_files.sort(key=lambda x: int(x.split(".")[1]))
    logger.debug(log_files)

    for log_file in log_files:
        logger.debug(f"Processing {log_file}")
        log_path = os.path.join(data_path, log_file)
        
        # execute bash script extract_gradients.sh
        os.system(f"bash {command_path} {log_path} >> {output_path}")
        