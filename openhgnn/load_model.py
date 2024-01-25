# load_model.py
# Author: Almas (I made this)

import argparse
from openhgnn.experiment import Experiment

def load_and_print_model(model_name, dataset_name, task_name, gpu, checkpoint_path):
    experiment = Experiment(model=model_name, dataset=dataset_name, task=task_name, gpu=gpu)

    # Load the model from the checkpoint
    experiment.load_model(checkpoint_path)

    # Print the loaded model
    print("Model loaded from checkpoint:")
    print(experiment.model)  # Assuming the model has a __str__ or __repr__ method

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='RGCN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')
    parser.add_argument('--dataset', '-d', default='acm4GTN', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means CPU')
    parser.add_argument('--checkpoint_path', '-c', required=True, type=str, help='path to the checkpoint file')
    args = parser.parse_args()

    load_and_print_model(args.model, args.dataset, args.task, args.gpu, args.checkpoint_path)
