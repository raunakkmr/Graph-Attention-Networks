import argparse
import json
import sys

import torch
import torch.nn as nn

from datasets import node_classification

def get_criterion(task):
    """
    Parameters
    ----------
    task : str
        Name of the task.

    Returns
    -------
    criterion : torch.nn.modules._Loss
        Loss function for the task.
    """
    if task == 'node_classification':
        criterion = nn.CrossEntropyLoss()

    return criterion

def get_dataset(args):
    """
    Parameters
    ----------
    args : tuple
        Tuple of task, dataset name and other arguments required by the dataset constructor.

    Returns
    -------
    dataset : torch.utils.data.Dataset
        The dataset.
    """
    task, dataset_name, *dataset_args = args
    if task == 'node_classification':
        if dataset_name == 'cora':
            dataset = node_classification.Cora(*dataset_args)

    return dataset

def get_fname(config):
    """
    Parameters
    ----------
    config : dict
        A dictionary with all the arguments and flags.

    Returns
    -------
    fname : str
        The filename for the saved model.
    """
    hidden_dims_str = '_'.join([str(x) for x in config['hidden_dims']])
    num_heads_str = '_'.join([str(x) for x in config['num_heads']])
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']
    dropout = config['dropout']
    transductive = str(config['transductive'])
    fname = 'gat_hidden_dims_{}_num_heads_{}_batch_size_{}_epochs_{}_lr_{}_weight_decay_{}_dropout_{}_transductive_{}.pth'.format(
        hidden_dims_str, num_heads_str, batch_size, epochs, lr,
        weight_decay, dropout, transductive)

    return fname

def parse_args():
    """
    Returns
    -------
    config : dict
        A dictionary with the required arguments and flags.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--json', type=str, default='config.json',
                        help='path to json file with arguments, default: config.json')

    parser.add_argument('--print_every', type=int, default=16,
                        help='print loss and accuracy after how many batches, default: 16')

    parser.add_argument('--dataset', type=str, choices=['cora'], default='cora',
                        help='name of the dataset, default=cora')
    parser.add_argument('--dataset_path', type=str,
                        default='/Users/raunak/Documents/Datasets/Cora', 
                        help='path to dataset')
    parser.add_argument('--self_loop', action='store_true',
                        help='whether to add self loops to adjacency matrix, default=False')
    parser.add_argument('--normalize_adj', action='store_true',
                        help='whether to normalize adj like in gcn, default=False')
    parser.add_argument('--transductive', action='store_true',
                        help='whether to use all nodes while training, default=False')

    parser.add_argument('--task', type=str,
                        choices=['unsupervised', 'node_classification'],
                        default='node_classification',
                        help='type of task, default=node_classification')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout parameter, default=0.5.')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use GPU, default: False')
    parser.add_argument('--hidden_dims', type=int, nargs="*",
                        help='dimensions of hidden layers, specify through config.json')
    parser.add_argument('--num_heads', type=int, nargs="*",
                        help='number of attention heads in each layer, length should be equal to len(hidden_dims)+1, specify through config.json')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='training batch size, default=8')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs, default=10')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate, default=1e-3')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay, default=5e-4')

    parser.add_argument('--save', action='store_true',
                        help='whether to save model in trained_models/ directory, default: False')
    parser.add_argument('--load', action='store_true',
                        help='whether to load model in trained_models/ directory')

    args = parser.parse_args()
    config = vars(args)
    if config['json']:
        with open(config['json']) as f:
            json_dict = json.load(f)
            config.update(json_dict)

    config['num_layers'] = len(config['hidden_dims']) + 1

    print('--------------------------------')
    print('Config:')
    for (k, v) in config.items():
        print("    '{}': '{}'".format(k, v))
    print('--------------------------------')

    return config