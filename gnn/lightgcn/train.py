import argparse
import os

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from gat import GAT

from recommenders.models.deeprec.deeprec_utils import prepare_hparams


def train(args: argparse.Namespace, model_type, data):

    
    hparams = prepare_hparams(
        yaml_file=args.hparams_yaml_file,
        MODEL_DIR=os.path.join('./gnn/lightgcn/outputs', args.dataset_size),
        epochs=args.train_epochs,
    )
    if model_type == 'lightgcn':
        model = LightGCN(hparams=hparams, data=data, seed=args.seed)
    elif model_type == 'gat':
        model = GAT(hparams=hparams, data=data, seed=args.seed)
    else:
        raise ValueError(f'unknown model type: {model_type}')
    
    if args.load_epoch:
        model.load(model_path=os.path.join(
            'gnn', 'lightgcn', 'outputs', args.dataset_size, f'epoch_{args.load_epoch}'
        ))
    else:
        print(f'training {model_type}')
        with Timer() as train_time:
            model.fit()
        print(f'Total training time: {train_time.interval} seconds')
    
    return model
