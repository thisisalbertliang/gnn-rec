from typing import Dict
import matplotlib.pylab as plt
import numpy as np
import os
import pickle

def load_metrics(metrics_dir):
    metrics_files = [f for f in os.listdir(metrics_dir) if os.path.isfile(os.path.join(metrics_dir, f)) and f.endswith(".metrics")]
    metrics = {}
    for file_name in metrics_files:
        with open(os.path.join(metrics_dir, file_name), 'rb') as f:
            metrics[file_name.strip(".metrics")] = pickle.load(f)
    return metrics

def plot(metrics: Dict[str, Dict[str, float]], save_path=None):
    labels = ['MAP @ K', 'nDCG @ K', 'Precision @ K', 'Recall @ K']
    
    x = np.arange(len(labels)) * 3
    width = 0.35
    delta = np.linspace(start=-width, stop=width, num=len(metrics))
    
    fig, ax = plt.subplots()
    rects = []
    for i, (model_name, model_metrics) in enumerate(metrics.items()):
        metrics_values = [
            model_metrics['map'], model_metrics['ndcg'], model_metrics['precision'], model_metrics['recall']
        ]
        rects.append(
            ax.bar(
                x + 2.5 * delta[i],
                metrics_values,
                width,
                label=model_name
            )
        )
    
    ax.set_ylabel('Scores')
    ax.set_title('Rank-based Metrics by Model')
    ax.set_xticks(x, labels)
    ax.legend()
    
    # for rect in rects:
    #     ax.bar_label(rect, padding=3)
    
    fig.tight_layout()
    
    # epochs = params.load_epoch if params.load_epoch else params.epochs
    # plot_filepath = os.path.join(
    #     'gnn', 'lightgcn', 'outputs', params.dataset_size, 'eval', f'metrics_epoch_{epochs}.png'
    # )
    if save_path is not None:
        fig.savefig(save_path)

if __name__ == "__main__":
    METRICS_DIR = 'gnn/outputs/100k/metrics'
    PLOT_DIR = 'gnn/outputs/plots'
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    metrics = load_metrics(METRICS_DIR)
    plot(metrics, save_path=f'{PLOT_DIR}/metrics.png')
