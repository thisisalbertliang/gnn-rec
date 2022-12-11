from typing import Dict
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
from collections import defaultdict
from heapq import heappush, heappop
from copy import deepcopy


_METRIC_DISPLAY_NAMES = ['Precision @ K', 'Recall @ K', 'nDCG @ K', 'MAP @ K']
_METRIC_NAMES = ['precision', 'recall', 'ndcg', 'map']
_METRIC_2_IDX = {name: idx for idx, name in enumerate(_METRIC_NAMES)}

_LABEL_FMT = '%.3f'
_LABEL_SIZE = 7

_BASELINES = ['SVD', 'NMF', 'SlopeOne', 'KNNBa', 'NormalPredictor']


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
    metric_2_best_baseline = dict()
    for model_name, metrics in model_2_metrics.items():
        if model_name in _BASELINES:
            for metric_name, metric_value in metrics.items():
                metric_2_best_baseline[metric_name] = max(
                    metric_2_best_baseline.get(metric_name, (-float('inf'), None)),
                    (metric_value, model_name)
                )
    return {
        metric_name: baseline
        for metric_name, (_, baseline) in metric_2_best_baseline.items()
    }


def _ensure_and_get_unique_epoch(model_2_metrics: Dict[str, Dict[str, float]]):
    """
    Ensures all models were trained for the same number of epochs
    and get this unique epoch
    """
    unique_epochs = set(metrics['epochs'] for metrics in model_2_metrics.values())
    assert len(unique_epochs) == 1
    unique_epochs = unique_epochs.pop()
    # delete epoch key from metrics
    for model in model_2_metrics:
        unqiue_epoch = model_2_metrics[model].pop('epochs')
    return unqiue_epoch


def plot_best_models_and_baselines(
    model_2_metrics: Dict[str, Dict[str, float]],
    save_path=None
):
    model_2_metrics = deepcopy(model_2_metrics) # defensive copy
    unique_epochs = _ensure_and_get_unique_epoch(model_2_metrics)
    
    x = np.arange(len(_METRIC_NAMES)) * 9  # the metric locations
    metric_width = 3  # the width of each metric
    bar_width = 1
    
    metric_2_best_model = get_top_models(model_2_metrics, n=1)
    best_models = set([
        model_name for _, [model_name] in metric_2_best_model.items()
    ])
    metric_2_all_best_models = defaultdict(list)
    for best_model in best_models:
        for metric_name in _METRIC_NAMES:
            metric_2_all_best_models[metric_name].append(
                (model_2_metrics[best_model][metric_name], best_model)
            )
    metric_2_sorted_models = {
        metric_name: [model for _, model in sorted(models, key=lambda x: x[0], reverse=True)]
        for metric_name, models in metric_2_all_best_models.items()
    }
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    delta = np.linspace(-metric_width, metric_width, len(best_models) + 2)
    # plot the best models
    colors = ['tab:red', 'tab:orange', 'tab:blue']
    for i, best_model in enumerate(best_models):
        ax.bar_label(fmt=_LABEL_FMT, size=_LABEL_SIZE, container=ax.bar(
            x + np.array([delta[metric_2_sorted_models[metric_name].index(best_model)] for metric_name in _METRIC_NAMES]),
            [model_2_metrics[best_model][metric_name] for metric_name in _METRIC_NAMES],
            width=bar_width,
            label=best_model[:-2],
            color=colors[i]
        ))
    # plot LightGCN
    ax.bar_label(fmt=_LABEL_FMT, size=_LABEL_SIZE, container=ax.bar(
        x + delta[len(best_models)],
        [model_2_metrics['degree_norm|direct|mean'][name] for name in _METRIC_NAMES],
        width=bar_width,
        label='LightGCN',
        color='wheat'
    ))
    # plot the best baseline
    metric_2_best_baseline = get_best_baseline(model_2_metrics)
    best_baseline_2_metrics = defaultdict(dict)
    for metric_name, best_baseline in metric_2_best_baseline.items():
        best_baseline_2_metrics[best_baseline][metric_name] = model_2_metrics[best_baseline][metric_name]
    colors = ['lightgrey', 'sandybrown']
    for i, (best_baseline, metrics) in enumerate(best_baseline_2_metrics.items()):
        ax.bar_label(fmt=_LABEL_FMT, size=_LABEL_SIZE, container=ax.bar(
            np.array([x[_METRIC_2_IDX[metric_name]] for metric_name in metrics.keys()]) + delta[len(best_models) + 1],
            [metric_value for metric_value in metrics.values()],
            width=bar_width,
            label=best_baseline if best_baseline != 'KNNBa' else 'KNN',
            color=colors[i]
        ))
    
    # ax.bar_label(rects)
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Rank-based Metrics of the Best GNNs\nbenchmarked by LightGCN and the Best Baseline Model\n(Epochs = {unique_epochs})')
    ax.set_xticks(x, _METRIC_DISPLAY_NAMES)
    ax.legend(loc='upper right', bbox_to_anchor=(1.065, 1.0))
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def plot_top_models_and_baselines(
    model_2_metrics: Dict[str, Dict[str, float]],
    n = 3,
    save_path=None
):
    """Plots the top N GNNs, LightGCN, and the best baseline for each metric"""
    model_2_metrics = deepcopy(model_2_metrics) # defensive copy

    x = np.arange(len(_METRIC_NAMES)) * 7  # the metric locations
    width = (n + 2) / 2  # the width of each metric
    delta = np.linspace(-width, width, n + 2)
    bar_width = 1
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    
    unique_epochs = _ensure_and_get_unique_epoch(model_2_metrics)

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
            np.array([x[_METRIC_2_IDX[metric_name]] for metric_name in metrics.keys()]) + np.array([delta[rank] for _, rank in metrics.values()]),
            [metric_value for metric_value, _ in metrics.values()],
            width=bar_width,
            label=model[:-2],
        )
    
    # plot LightGCN performance
    ax.bar(
        x + delta[n],
        [model_2_metrics['degree_norm|direct|mean'][name] for name in _METRIC_NAMES],
        width=bar_width,
        label='LightGCN'
    )
    
    # plot best baseline performance
    for best_baseline, metrics in best_baseline_2_metrics.items():
        ax.bar(
            np.array([x[_METRIC_2_IDX[metric_name]] for metric_name in metrics.keys()]) + delta[n + 1],
            [metric_value for metric_value in metrics.values()],
            width=bar_width,
            label=best_baseline if best_baseline != 'KNNBa' else 'KNN'
        )
    
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Rank-based Metrics of Top Performing GNNs\nbenchmarked by LightGCN and the Best Baseline Model\n(Epochs = {unique_epochs})')
    ax.set_xticks(x, _METRIC_DISPLAY_NAMES)
    ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def get_architectures(component: str = 'info_updater', seed: int = 0):
    """
    Returns a list of architectures for the given component, where each architecture
    is a tuple of the form (architecture, algorithm)
    """
    algorithms = {
        'neighbor_aggregator': ['degree_norm', 'attention'],
        'info_updater': ['direct', 'single_linear', 'multi_linear'],
        'final_node_repr': ['mean', 'concat', 'weighted', 'attention']
    }
    if component == 'neighbor_aggregator':
        architectures = [
            (f'{algo}|direct|mean_{seed}', algo)
            for algo in algorithms[component]
        ]
    elif component == 'info_updater':
        architectures = [
            (f'degree_norm|{algo}|mean_{seed}', algo)
            for algo in algorithms[component]
        ]
    elif component == 'final_node_repr':
        architectures = [
            (f'degree_norm|direct|{algo}_{seed}', algo)
            for algo in algorithms[component]
        ]
    else:
        raise ValueError(f'Invalid component: {component}')

    return architectures


def plot_metrics_by_algorithms(
    model_2_metrics: Dict[str, Dict[str, float]],
    component: str = 'info_updater',
    save_path=None
):
    model_2_metrics = deepcopy(model_2_metrics) # defensive copy
    
    architectures = get_architectures(component)
    
    component_display_names = {
        'neighbor_aggregator': 'Neighbor Aggregator',
        'info_updater': 'Information Update',
        'final_node_repr': 'Final Node Representation'
    }
    
    x = np.arange(len(_METRIC_NAMES)) * 8  # the metric locations
    metric_width = len(architectures) / 2  # the width of each metric
    bar_width = 1  # the width of each bar
    delta = np.linspace(-metric_width, metric_width, len(architectures))
    
    unique_epochs = _ensure_and_get_unique_epoch(model_2_metrics)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    
    for i, (architecture, algorithm) in enumerate(architectures):
        metrics = model_2_metrics[architecture]
        ax.bar_label(fmt=_LABEL_FMT, size=_LABEL_SIZE, container=ax.bar(
            x + delta[i],
            [metrics[name] for name in _METRIC_NAMES],
            width=bar_width,
            label=algorithm
        ))
    
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Rank-based Metrics of Algorithms in {component_display_names[component]}\n(Epochs = {unique_epochs})')
    ax.set_xticks(x, _METRIC_DISPLAY_NAMES)
    ax.legend(loc='upper right')
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    METRICS_DIR = 'gnn/outputs/100k/metrics'
    PLOT_DIR = 'gnn/outputs/100k/plots'
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    model_2_metrics = load_metrics(METRICS_DIR)
    # plot_top_models_and_baselines(model_2_metrics, save_path=f'{PLOT_DIR}/ranking_metrics.png', n=1)
    plot_metrics_by_algorithms(model_2_metrics, component='neighbor_aggregator', save_path=f'{PLOT_DIR}/neighbor_aggregator.png')
    plot_metrics_by_algorithms(model_2_metrics, component='info_updater', save_path=f'{PLOT_DIR}/info_updater.png')
    plot_metrics_by_algorithms(model_2_metrics, component='final_node_repr', save_path=f'{PLOT_DIR}/final_node_repr.png')
    plot_best_models_and_baselines(model_2_metrics, save_path=f'{PLOT_DIR}/best_models_and_baselines.png')
