import argparse
import os


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # parser.add_argument(
    #     '--dataset-size',
    #     choices=['100k', '1m', '10m', '20m'],
    #     default='100k',
    #     type=str,
    # )
    # parser.add_argument(
    #     '--train-size',
    #     default=0.75,
    #     type=float,
    # )
    # parser.add_argument(
    #     '--seed',
    #     default=10708,
    #     type=int,
    # )
    parser.add_argument(
        '--hparams-yaml-file',
        default=os.path.join(
            'gnn', 'configs', 'lightgcn_hparams.yaml'
        ),
        type=str,
    )
    parser.add_argument(
        '--load-epoch',
        default=0,
        type=int,
    )
    # parser.add_argument(
    #     '--top-k',
    #     default=10,
    #     type=int,
    # )
    # parser.add_argument(
    #     '--train-epochs',
    #     default=50,
    #     type=int
    # )
    
    return parser.parse_args()
