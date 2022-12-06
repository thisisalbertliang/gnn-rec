from args import parse_args
from train import train
from eval import eval
from plot import plot

from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split

if __name__ == '__main__':
    
    args = parse_args()

    dataset = movielens.load_pandas_df(size=args.dataset_size)
    train_set, test_set = python_stratified_split(data=dataset, ratio=args.train_size)
    data = ImplicitCF(train=train_set, test=test_set, seed=args.seed)

    models = {
        'lightgcn': train(args, 'lightgcn', data),
        'gat': train(args, 'gat', data)
    }
    metrics = eval(args, models, test_set)
    plot(args, metrics)
