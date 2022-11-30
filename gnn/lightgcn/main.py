from args import parse_args
from train import train
from eval import eval
from plot import plot


if __name__ == '__main__':
    
    args = parse_args()

    model, test = train(args)
    metrics = eval(args, model, test)
    plot(args, metrics)
