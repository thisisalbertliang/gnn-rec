import os

from recommenders.utils.timer import Timer
from gnn import GNN


def train(params, data):
    model = GNN(params=params, data=data, seed=params.seed)
    
    if params.load_epoch:
        model_load_path = os.path.join(
            params.model_dir, params.dataset_size, "checkpoints", params.model_name, f'epoch_{params.load_epoch}'
        )
        model.load(model_path=model_load_path)
        print(f'>>> loaded {params.model_name} from {model_load_path}')
    else:
        print(f'>>> training {params.model_name}')
        with Timer() as train_time:
            model.fit()
        print(f'total training time: {train_time.interval} seconds')
    
    return model
