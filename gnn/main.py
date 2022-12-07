from args import parse_args
from train import train
from eval import eval
from plot import plot

from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split


if __name__ == '__main__':
    
    args = parse_args()

    params = prepare_hparams(
        yaml_file=args.hparams_yaml_file,
        load_epoch=args.load_epoch
    )
    params.model_name = f'{params.neighbor_aggregator}|{params.info_updater}|{params.final_node_repr}'

    dataset = movielens.load_pandas_df(size=params.dataset_size)
    train_set, test_set = python_stratified_split(data=dataset, ratio=params.trainset_ratio)
    data = ImplicitCF(train=train_set, test=test_set, seed=params.seed)

    models = {
        params.model_name : train(params, data)
        # 'LightGCN': train(params, 'lightgcn', data),
        # 'GAT': train(params, 'gat', data)
    }

    metrics = eval(params, models, test_set)
    # plot(params, metrics)
