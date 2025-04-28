import torch
import os
from schnetpack.datasets import QM9
import schnetpack.transform as trn

from tutorials.utils import set_data_prefix

# Example command to run the script from within schnetpack directory:
"""
python -m tutorials.QM9.inference
"""

if __name__=="__main__":
    ################### Set up the data ###################
    qm9tut = os.path.expanduser('~/whk/code/schnetpack/tutorials/QM9/output')

    os.makedirs(qm9tut, exist_ok=True)

    data_prefix = set_data_prefix()

    # Load data from QM9
    qm9data = QM9(
        f'{data_prefix}/qm9.db',
        batch_size=100,
        num_train=1000,
        num_val=1000,
        transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
            trn.CastTo32()
        ],
        property_units={QM9.U0: 'eV'},
        num_workers=1,
        split_file=os.path.join(qm9tut, "split.npz"),
        pin_memory=True, # set to false, when not using a GPU
        load_properties=[QM9.U0], #only load U0 property
    )
    qm9data.prepare_data()
    qm9data.setup()

    ################### Inference ###################
    best_model = torch.load(os.path.join(qm9tut, 'best_inference_model'), map_location="cpu")

    for batch in qm9data.test_dataloader():
        result = best_model(batch)
        print("Result dictionary:", result)
        break