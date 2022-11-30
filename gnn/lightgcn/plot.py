from typing import Dict
import matplotlib.pylab as plt
import numpy as np
import os
import argparse


def plot(args: argparse.Namespace, metrics: Dict[str, Dict[str, float]]):
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
    
    epochs = args.load_epoch if args.load_epoch else args.train_epochs
    plot_filepath = os.path.join(
        'gnn', 'lightgcn', 'outputs', args.dataset_size, 'eval', f'metrics_epoch_{epochs}.png'
    )
    fig.savefig(plot_filepath)
