from typing import Dict
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
from collections import defaultdict
from heapq import heappush, heappop


def load_metrics(metrics_dir):
    metrics_files = [f for f in os.listdir(metrics_dir) if os.path.isfile(os.path.join(metrics_dir, f)) and f.endswith(".metrics")]
    model_2_metrics = {}
    for file_name in metrics_files:
        with open(os.path.join(metrics_dir, file_name), 'rb') as f:
            model_2_metrics[file_name.strip(".metrics")] = pickle.load(f)
    return model_2_metrics


def get_top_models(model_2_metrics: Dict[str, Dict[str, float]], n: int = 3):
    """
    Returns a dict mapping metric names to the top n models for 
    the metric sorted by metric value in descending order
    """
    metric_2_top_models = defaultdict(list)

    for model_name, metrics in model_2_metrics.items():
        for metric_name, metric_value in metrics.items():
            top_models = metric_2_top_models[metric_name]

            heappush(top_models, (metric_value, model_name))
            if len(top_models) > n:
                heappop(top_models)
    
    for metric_name, top_models in metric_2_top_models.items():
        metric_2_top_models[metric_name] = [
            model_name for _, model_name in sorted(top_models, key=lambda x: x[0], reverse=True)
        ]
    
    return metric_2_top_models


def get_best_baseline(model_2_metrics: Dict[str, Dict[str, float]]):
    """
    Returns a dict mapping from metric names to the best baseline model
    """
    BASELINES = ['SVD', 'NMF', 'SlopeOne', 'KNNBasic', 'NormalPredictor']
    metric_2_best_baseline = dict()
    for model_name, metrics in model_2_metrics.items():
        if model_name in BASELINES:
            for metric_name, metric_value in metrics.items():
                metric_2_best_baseline[metric_name] = max(
                    metric_2_best_baseline.get(metric_name, (-float('inf'), None)),
                    (metric_value, model_name)
                )
    return {
        metric_name: baseline
        for metric_name, (_, baseline) in metric_2_best_baseline.items()
    }


def plot_top_models_and_baselines(
    model_2_metrics: Dict[str, Dict[str, float]],
    n = 3,
    save_path=None
):
    """Plots the top N GNNs, LightGCN, and the best baseline for each metric"""
    metric_display_names = ['Precision @ K', 'Recall @ K', 'nDCG @ K', 'MAP @ K']
    metric_names = ['precision', 'recall', 'ndcg', 'map']

    x = np.arange(len(metric_display_names)) * 10  # the metric locations
    width = 3  # the width of each metric
    delta = np.linspace(-width, width, n + 2)
    bar_width = 1
    
    metric_2_idx = {metric: i for i, metric in enumerate(metric_names)}
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    
    # ensures all models were trained for the same number of epochs
    unique_epochs = set(model['epochs'] for model in model_2_metrics.values())
    assert len(unique_epochs) == 1
    unique_epochs = unique_epochs.pop()
    # delete epoch key from metrics
    for model in model_2_metrics:
        model_2_metrics[model].pop('epochs')

    metric_2_top_models = get_top_models(model_2_metrics, n=n)
    top_model_2_metrics = defaultdict(dict)
    for metric_name, top_models in metric_2_top_models.items():
        for rank, model in enumerate(top_models):
            top_model_2_metrics[model][metric_name] = (model_2_metrics[model][metric_name], rank)
    
    metric_2_best_baseline = get_best_baseline(model_2_metrics)
    best_baseline_2_metrics = defaultdict(dict)
    for metric_name, best_baseline in metric_2_best_baseline.items():
        best_baseline_2_metrics[best_baseline][metric_name] = model_2_metrics[best_baseline][metric_name]
    
    # plot the top N GNNs performance
    for model, metrics in top_model_2_metrics.items():
        ax.bar(
            np.array([x[metric_2_idx[metric_name]] for metric_name in metrics.keys()]) + np.array([delta[rank] for _, rank in metrics.values()]),
            [metric_value for metric_value, _ in metrics.values()],
            width=bar_width,
            label=model,
        )
    
    # plot LightGCN performance
    ax.bar(
        x + delta[n],
        [model_2_metrics['degree_norm|direct|mean'][name] for name in metric_names],
        width=bar_width,
        label='LightGCN'
    )
    
    #plot best baseline performance
    for best_baseline, metrics in best_baseline_2_metrics.items():
        ax.bar(
            np.array([x[metric_2_idx[metric_name]] for metric_name in metrics.keys()]) + delta[n + 1],
            [metric_value for metric_value in metrics.values()],
            width=bar_width,
            label=best_baseline
        )
    
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Rank-based Metrics of Top Performing GNNs\nbenchmarked by LightGCN and the Best Baseline Model\n(Epochs = {unique_epochs})')
    ax.set_xticks(x, metric_display_names)
    ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)

if __name__ == "__main__":
    METRICS_DIR = 'gnn/outputs/100k/metrics'
    PLOT_DIR = 'gnn/outputs/100k/plots'
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    model_2_metrics = load_metrics(METRICS_DIR)
    plot_top_models_and_baselines(model_2_metrics, save_path=f'{PLOT_DIR}/ranking_metrics.png')
