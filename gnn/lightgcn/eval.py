import pandas as pd
import argparse
from tqdm import tqdm
from heapq import heappush, heappop
import os
import pickle

from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
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


def recommend_k_items(args: argparse.Namespace, algo: surprise.AlgoBase, test: pd.DataFrame, top_k: int):
    trainset = Dataset.load_builtin(f'ml-{args.dataset_size}').build_full_trainset()
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


def eval(args: argparse.Namespace, models, test: pd.DataFrame):
    epochs = args.load_epoch if args.load_epoch else args.train_epochs
    metrics_filepath = os.path.join(
        'gnn', 'lightgcn', 'outputs', args.dataset_size, 'eval', f'metrics_epoch_{epochs}.pt'
    )
    # if os.path.exists(metrics_filepath):
    #     with open(metrics_filepath, 'rb') as f:
    #         return pickle.load(f)
    
    eval_results = {
        'GAT': models['gat'].recommend_k_items(test, top_k=args.top_k, remove_seen=False),
        'LightGCN': models['lightgcn'].recommend_k_items(test, top_k=args.top_k, remove_seen=False)
    }
    
    baselines = (
        ('SVD', SVD(random_state=0)),
        ('NMF', NMF(random_state=0)),
        ('SlopeOne', SlopeOne()),
        ('KNNBasic', KNNBasic()),
        ('NormalPredictor', NormalPredictor()),
    )
    for name, algo in baselines:
        eval_results[name] = recommend_k_items(args, algo=algo, test=test, top_k=args.top_k)
    
    metrics = dict()
    for name, topk_scores in eval_results.items():
        eval_map = map_at_k(test, topk_scores, k=args.top_k)
        eval_ndcg = ndcg_at_k(test, topk_scores, k=args.top_k)
        eval_precision = precision_at_k(test, topk_scores, k=args.top_k)
        eval_recall = recall_at_k(test, topk_scores, k=args.top_k)

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
            'map': eval_map, 'ndcg': eval_ndcg, 'precision': eval_precision, 'recall': eval_recall
        }

    os.makedirs(
        os.path.dirname(metrics_filepath),
        exist_ok=True
    )
    with open(file=metrics_filepath, mode='wb') as f:
        pickle.dump(metrics, f)
    
    return metrics
