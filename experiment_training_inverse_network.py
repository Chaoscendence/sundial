import numpy as np
import pandas as pd
import os 
import torch
import logging
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.utils.data as data
from sundial.utils import save_checkpoint
from collections import Counter
from sundial.neural_models.piston_network import InverseNetwork
from sundial.physical_models.ellipsometry import compute_rt_from_nkz
from sundial.utils import get_logger
from sundial.training import train as train_model
from sundial.training import set_config
from sundial.ellipsometric_dataset import Ellipsometry
from sundial.optical_properties_dataset import ReflectivityTransmission
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sundial.utils import set_my_favorite_plot_settings
from sundial.utils import smooth
set_my_favorite_plot_settings()

import argparse
parser = argparse.ArgumentParser(description='Apply AI to optic science')
parser.add_argument('--epochs', default=100, help='number of epochs', type=int)
parser.add_argument('--batch_size', default=32, help='batch size', type=int)
parser.add_argument('--train', dest='train', action='store_true', 
                    help='flag to start training')
parser.set_defaults(train=False)
parser.add_argument('--test', dest='test', action='store_true', help='run test')
parser.set_defaults(test=False)
parser.add_argument('--valid', dest='valid', action='store_true', 
                    help='run valid')
parser.set_defaults(valid=False)
parser.add_argument('--eval', dest='eval', action='store_true', 
                    help='run series of tests to evaluate performance')
parser.set_defaults(eval=False)
parser.add_argument('--plot', dest='plot', action='store_true', help='Plot')
parser.set_defaults(plot=False)
parser.add_argument('--cpu', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)
parser.add_argument('--models_dir', dest="models_dir", type=str)
parser.set_defaults(models_dir="TrainedModels")
parser.add_argument('--log', dest="log", type=str)
parser.set_defaults(log="main.log")
parser.add_argument('--dataset', dest="dataset", type=str)
parser.set_defaults(dataset="Ellisometry")
parser.add_argument('--dataset_resplitting', dest='dataset_resplitting', 
                    action='store_true', 
                    help='flag to resplit dataset to training and test sets')
parser.set_defaults(dataset_resplitting=False)
parser.add_argument('--data_augmentation', dest='data_augmentation', 
                    action='store_true', 
                    help='use data augmentation or not.')
parser.set_defaults(data_augmentation=False)
parser.add_argument('--seed', default=0, help='random seed', type=int)
args = parser.parse_args()


################################################################################
# GLOBAL SETTINGS
settings = {
    'model': None,
    'dataset': None,
    'dataset_name': args.dataset,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'log_dir': args.log,
    'logger': None,
    'cuda': args.cuda,
    'models_dir': args.models_dir,
    'data_augmentation': args.data_augmentation,
    'random_seed': args.seed    
}

# Dataset
if args.dataset.lower() == "ellisometry":
    settings['dataset_name'] = "Ellisometry"
    settings['dataset'] = Ellipsometry(data_type="with_loss")
else:
    print('Not exist')

# Methods
lenet_settings = {
    'ConvLayers': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
                   256, 256, 'U', 128, 128, 'U', 64, 64, 'U'],
    'InChannels': 2,
    'OutChannels': 10,
}
net = InverseNetwork(settings=lenet_settings)
folder_name = "trained_models/inverse_networks/"
subfolder_name = "natural_materials_inverse_HG_MSELoss_OnlyRT"

settings['model'] = net
settings['epochs'] = 1000
settings['batch_size'] = 64
settings['num_valid'] = 30
settings['num_test'] = 30
settings['augmentation'] = False
settings['models_dir'] = os.path.join(folder_name, subfolder_name)
if not os.path.exists(settings['models_dir']):
    os.makedirs(settings['models_dir'])
settings['log'] = os.path.join(folder_name, subfolder_name+'.log') 
logger = get_logger(log_file_name=settings['log'], 
                    logging_level=logging.INFO)
settings['logger'] = logger
settings['optimizer'] = torch.optim.Adam(net.parameters(), lr=1e-4, 
                                         weight_decay=1e-5)    
settings['scheduler'] = lr_scheduler.CosineAnnealingLR(settings['optimizer'], 
                                                       T_max=200, eta_min=1e-5)


################################################################################
def main():

    #--------------------------------------------------------------------------
    # RT
    """
    ds_rt = ReflectivityTransmission(data_type="with_loss")
    X_train_rt, X_valid_rt, X_test_rt, Y_train_rt, Y_valid_rt, Y_test_rt = \
        ds_rt.train_valid_test_split_randomly(
            num_valid=settings['num_valid'], 
            num_test=settings['num_test'], 
            augmentation=settings['augmentation'],
            shuffle=True, 
            random_seed=settings['random_seed'],
            task='inverse')

    X_train = X_train_rt
    X_valid = X_valid_rt
    X_test = X_test_rt
    Y_train = Y_train_rt
    Y_valid = Y_valid_rt
    Y_test = Y_test_rt
    """
    #--------------------------------------------------------------------------
    ds_dp = Ellipsometry(data_type="with_loss")
    X_train_dp, X_valid_dp, X_test_dp, Y_train_dp, Y_valid_dp, Y_test_dp = \
        ds_dp.train_valid_test_split_randomly(
            num_valid=settings['num_valid'], 
            num_test=settings['num_test'], 
            augmentation=settings['augmentation'],
            shuffle=True, 
            random_seed=settings['random_seed'],
            task='inverse')

    X_train = X_train_dp
    X_valid = X_valid_dp
    X_test = X_test_dp
    Y_train = Y_train_dp
    Y_valid = Y_valid_dp
    Y_test = Y_test_dp

    print(X_train.shape, X_valid.shape, X_test.shape, 
          Y_train.shape, Y_valid.shape, Y_test.shape)

    if args.train:
        train_model([X_train, X_valid, X_test, Y_train, Y_valid, Y_test])
        return 


if __name__ == "__main__":

    logger.info(settings)
    set_config(settings)
    main()
