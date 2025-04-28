
from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList

from tutorials.utils import set_data_prefix

# Example command to run the script:
"""
python -m tutorials.data_loading.predefined_datasets
"""


if __name__=="__main__":
    print("Hello, world!")

    # Set data prefix
    data_prefix = set_data_prefix()
    print('Data prefix:', data_prefix)

    # import data
    qm9data = QM9(
        f'{data_prefix}/qm9.db',
        batch_size=10,
        num_train=110000,
        num_val=10000,
        split_file=f'{data_prefix}/split_qm9.npz',
        transforms=[ASENeighborList(cutoff=5.)],
        )
    # prepare data
    qm9data.prepare_data()
    qm9data.setup()

    # print some information
    print('Number of reference calculations:', len(qm9data.dataset))
    print('Number of train data:', len(qm9data.train_dataset))
    print('Number of validation data:', len(qm9data.val_dataset))
    print('Number of test data:', len(qm9data.test_dataset))
    print('Available properties:')

    for p in qm9data.dataset.available_properties:
        print('-', p)

    # load first data point and print some information
    example = qm9data.dataset[0]
    print('Properties:')

    for k, v in example.items():
        print('-', k, ':', v.shape)
    
    # load first batch of data
    for batch in qm9data.val_dataloader():
        print(batch.keys())
        print('System index:', batch['_idx_m'])
        print('Center atom index:', batch['_idx_i'])
        print('Neighbor atom index:', batch['_idx_j'])
        print('Total energy at 0K:', batch[QM9.U0])
        print('HOMO:', batch[QM9.homo])
        break