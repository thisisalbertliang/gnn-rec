from args import parse_args
from train import train
from eval import eval

from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split

import pandas as pd
import numpy as np

import pprint
import itertools


if __name__ == '__main__':
    
    args = parse_args()

    params = prepare_hparams(
        yaml_file=args.hparams_yaml_file,
        load_epoch=args.load_epoch
    )

    dataset = movielens.load_pandas_df(size=params.dataset_size)

    num_seeds = 10
    original_seed = params.seed
    all_metrics = {}
    for k in params.metrics:
        all_metrics[k] = []
        all_metrics[k+'_std'] = []
    all_metrics['modelname'] = []

    for neighbor_aggregator in ['degree_norm', 'attention']:
        for info_updater in ['direct', 'single_linear', 'multi_linear']:
            for final_node_repr in ['mean', 'concat', 'weighted', 'attention']:

                params.neighbor_aggregator = neighbor_aggregator
                params.info_updater = info_updater
                params.final_node_repr = final_node_repr
                model_name = f'{neighbor_aggregator}|{info_updater}|{final_node_repr}'

                metrics_accumulator = {k: [] for k in params.metrics}
                for i in range(num_seeds):

                    params.model_name = f'{model_name}_{i}'
                    params.seed = original_seed + i

                    train_set, test_set = python_stratified_split(data=dataset, ratio=params.trainset_ratio, seed=params.seed)
                    data = ImplicitCF(train=train_set, test=test_set, seed=params.seed)

                    models = {
                        params.model_name : train(params, data)
                    }

                    metrics = eval(params, models, test_set, eval_baselines=False)[params.model_name]
                    # import pdb; pdb.set_trace()
                    for k in params.metrics:
                        metrics_accumulator[k].append(metrics[k])

                for k in params.metrics:
                    all_metrics[k].append(np.mean(metrics_accumulator[k]))
                    all_metrics[k+'_std'].append(np.std(metrics_accumulator[k]))
                all_metrics['modelname'].append(model_name)

    df = pd.DataFrame(data=all_metrics)
    df.to_csv(params.automl_res_path)
