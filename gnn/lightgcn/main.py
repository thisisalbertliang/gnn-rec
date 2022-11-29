import pickle
import os
import pandas as pd
import argparse
from heapq import heappush, heappop
from tqdm import tqdm

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.deeprec.deeprec_utils import prepare_hparams

import surprise
from surprise import (
    Dataset,
    KNNBasic
)

from args import parse_args
from train import train
from eval import eval


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


if __name__ == '__main__':
    
    args = parse_args()
        
    model, test = train(args)
    eval(args, model, test)
