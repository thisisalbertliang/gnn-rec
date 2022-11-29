import argparse
import os

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.models.deeprec.deeprec_utils import prepare_hparams


def train(args: argparse.Namespace):
    dataset = movielens.load_pandas_df(size=args.dataset_size)
    train, test = python_stratified_split(data=dataset, ratio=args.train_size)
    data = ImplicitCF(train=train, test=test, seed=args.seed)
    
    hparams = prepare_hparams(yaml_file=args.hparams_yaml_file)
    model = LightGCN(hparams=hparams, data=data, seed=args.seed)
    
    if args.load_epoch:
        model.load(model_path=os.path.join(
            'gnn', 'lightgcn', 'outputs', f'epoch_{args.load_epoch}'
        ))
    else:
        with Timer() as train_time:
            model.fit()
        print(f'Total training time: {train_time.interval} seconds')
    
    return model, test
