import pandas as pd
import argparse
from tqdm import tqdm
from heapq import heappush, heappop
import os
import pickle

from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

import surprise
from surprise import (
    Dataset,
    KNNBasic,
    NMF,
    NormalPredictor,
    SlopeOne,
    SVD,
)


def recommend_k_items(params, algo: surprise.AlgoBase, test: pd.DataFrame, top_k: int):
    trainset = Dataset.load_builtin(f'ml-{params.dataset_size}').build_full_trainset() # TODO: train on trainset only
    algo.fit(trainset=trainset)
    
    res = []
    for uid in tqdm(test.userID.unique()):
        min_heap = []
        for iid in test.itemID.unique():
            pred = algo.predict(uid=str(uid), iid=str(iid))
            
            heappush(min_heap, (pred.est, pred.uid, pred.iid))
            if len(min_heap) > top_k:
                heappop(min_heap)
        
        min_heap.sort(reverse=True)
        for est, uid, iid in min_heap:
            res.append([int(uid), int(iid), float(est)])
    
    res = pd.DataFrame(res, columns=['userID', 'itemID', 'prediction'])
    res = res.astype({'userID': 'int64', 'itemID': 'int64', 'prediction': 'float32'})
    return res

def evaluate_baselines(params, test):
    eval_results = {}
    baselines = (
        ('SVD', SVD(random_state=0)),
        ('NMF', NMF(random_state=0)),
        ('SlopeOne', SlopeOne()),
        ('KNNBasic', KNNBasic()),
        ('NormalPredictor', NormalPredictor()),
    )
    for name, algo in baselines:
        print(f'>>> running recommendation with {name}')
        eval_results[name] = recommend_k_items(params, algo=algo, test=test, top_k=params.top_k)
    return eval_results


def compute_metrics(params, test, eval_results, dump_dir=None):
    if dump_dir is not None:
        os.makedirs(
            os.path.dirname(dump_dir),
            exist_ok=True
        )

    metrics = dict()
    for name, topk_scores in eval_results.items():
        eval_map = map_at_k(test, topk_scores, k=params.top_k)
        eval_ndcg = ndcg_at_k(test, topk_scores, k=params.top_k)
        eval_precision = precision_at_k(test, topk_scores, k=params.top_k)
        eval_recall = recall_at_k(test, topk_scores, k=params.top_k)

        print(f'********************** {name} Results **********************')
        print(
            f"{name} MAP:\t{eval_map}",
            f"{name} NDCG:\t{eval_ndcg}",
            f"{name} Precision@K:\t{eval_precision}",
            f"{name} Recall@K:\t{eval_recall}",
            sep='\n'
        )
        print(f'************************************************************')
        
        metrics[name] = {
            'map': eval_map, 'ndcg': eval_ndcg, 'precision': eval_precision, 'recall': eval_recall,
            'epochs': params.epochs
        }

        with open(f'{dump_dir}/{name}.metrics', 'wb') as f:
            pickle.dump(metrics[name], f)
    return metrics


def eval(params, models, test: pd.DataFrame, eval_baselines=True):
    # epochs = params.load_epoch if params.load_epoch else params.epochs
    metrics_dir = os.path.join(params.model_dir, params.dataset_size, 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    eval_results = {}

    for name, model in models.items():
        print(f'>>> running recommendation with {name}')
        eval_results[name] = model.recommend_k_items(test, top_k=params.top_k, remove_seen=False)

    if eval_baselines:
        eval_results |= evaluate_baselines(params, test)
    
    return compute_metrics(params, test, eval_results, dump_dir=metrics_dir)
    