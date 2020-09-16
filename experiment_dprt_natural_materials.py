import numpy as np
import pandas as pd
import os
from os.path import join 
import shutil 
import torch
import logging
from scipy.interpolate import interp1d
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.utils.data as data
from sundial.utils import save_checkpoint
from collections import Counter
from sundial.utils import get_logger
from sundial.training import train as train_model
from sundial.training import set_config
from sundial.ellipsometric_dataset import Ellipsometry
from sundial.optical_properties_dataset import ReflectivityTransmission
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from sundial.physical_models.ellipsometry import compute_delta_psi_from_nkz
from sundial.physical_models.ellipsometry import compute_rt_from_nkz
from sundial.training_piston_network import forward_program
from sundial.utils import smooth
from sundial.neural_models.piston_network import ForwardNetwork
from sundial.neural_models.piston_network import InverseNetwork
from sundial.neural_models.piston_network import MIMONet
from sundial.neural_models.piston_network import ForwardNetBag
from sundial.neural_models.piston_network import InverseNetBag
from sundial.training_piston_network import PistonNetworkTrainer
from sundial.utils import set_my_favorite_plot_settings
# set_my_favorite_plot_settings()
import matplotlib.pyplot as plt
from sundial.utils import get_chinese_font
g_font_zh = get_chinese_font()
from meta_settings import *


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
parser.add_argument('--plot_d', dest='plot_d', action='store_true', help='Plot_D')
parser.set_defaults(plot_d=False)
parser.add_argument('--rand', dest='rand', action='store_true', help='Rand')
parser.set_defaults(rand=False)
parser.add_argument('--cpu', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)
parser.add_argument('--models_dir', dest="models_dir", type=str)
parser.set_defaults(models_dir="TrainedModels")
parser.add_argument('--log', dest="log", type=str)
parser.set_defaults(log="main.log")
parser.add_argument('--dataset', dest="dataset", type=str)
parser.set_defaults(dataset="Ellisometric")
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
import configparser

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


inverse_folder_dp = "trained_models/inverse_networks/" + \
            "natural_materials_inverse_HG_MSELoss_OnlyDP_2020"
forward_folder_dp = "trained_models/forward_networks/" + \
            "natural_materials_forward_HG_MSELoss_OnlyDP_2020"
inverse_folder_rt = "trained_models/inverse_networks/" + \
            "natural_materials_inverse_HG_MSELoss_OnlyRT_2020"
forward_folder_rt = "trained_models/forward_networks/" + \
            "natural_materials_forward_HG_MSELoss_OnlyRT_2020"

inverse_network_dp = torch.load(inverse_folder_dp
                                + "/best_single_net.tar")['net']
forward_network_dp = torch.load(forward_folder_dp
                                + "/best_single_net.tar")['net']
forward_network_rt = torch.load(forward_folder_rt 
                                + "/best_single_net.tar")['net']
inverse_network_rt = torch.load(inverse_folder_rt 
                                + "/best_single_net.tar")['net']

forward_networks = ForwardNetBag([forward_network_dp, forward_network_rt])
inverse_networks = InverseNetBag([inverse_network_dp, inverse_network_rt])

net = MIMONet()
net.initialize(inverse_network=inverse_networks, 
               forward_network=forward_networks)



real_materials = 'W'
elem_folder = real_materials + '_DprtLoss_Run2020_DPRT_MD0224_w0.5'
SAVED_FOLDER = "trained_models/NaturalMaterialsWithLoss/ForPistonTrainingWithRT"
SAVED_SUBFOLDER = "PistionNetwork_"+elem_folder

# Dataset
if args.dataset.lower() == "ellisometric":
    settings['dataset_name'] = "Ellisometric"
    settings['dataset'] = Ellipsometry(data_type="with_loss", 
                                       materials=real_materials)
else:
    print('Not exist')

settings['model'] = net
settings['epochs'] = 2000
settings['batch_size'] = 1
settings['num_valid'] = 1
settings['num_test'] = 1
settings['augmentation'] = False
settings['models_dir'] = os.path.join(SAVED_FOLDER, SAVED_SUBFOLDER)
settings['results_dir'] = os.path.join(SAVED_FOLDER, "ResultsByMaterials_" 
    + SAVED_SUBFOLDER)
settings['log'] = os.path.join(SAVED_FOLDER, SAVED_SUBFOLDER+'.log') 
logger = get_logger(log_file_name=settings['log'], logging_level=logging.INFO)
settings['logger'] = logger
settings['forward_optimizer'] = torch.optim.Adam(
    net.forward_network.parameters(), lr=1e-5, weight_decay=0)
settings['inverse_optimizer'] = torch.optim.Adam(
    net.inverse_network.parameters(), lr=1e-5, weight_decay=0)
if not os.path.exists(settings['models_dir']):
    os.makedirs(settings['models_dir'])
settings['forward_scheduler'] = lr_scheduler.CosineAnnealingLR(
    settings['forward_optimizer'], T_max=50, eta_min=1e-5)
settings['inverse_scheduler'] = lr_scheduler.CosineAnnealingLR(
    settings['inverse_optimizer'], T_max=50, eta_min=1e-5)

#-------------------------------------------------------------------------------
def program_predict(X):
    device = torch.device("cuda")
    # net = torch.load(os.path.join(settings['models_dir'], 
    #     'best_single_net.tar'))
    net = settings['model']
    # net = net['net']
    net.eval()
    net = net.to(device).float()        
    X_input = torch.from_numpy(X).to(device).float()
    top_nkz1z2_pred = net.inverse_net_forward(X_input)
    tnp = top_nkz1z2_pred.detach().cpu().numpy()
    tnp = tnp[:,0:4,:]
    Y_preds_program = forward_program(tnp)
    return Y_preds_program



################################################################################
def main():

    settings['dataset'] = ReflectivityTransmission(data_type="with_loss", 
                                                   materials=real_materials)
    ds = settings['dataset']
    X_train_rt, X_valid_rt, X_test_rt, Y_train_rt, Y_valid_rt, Y_test_rt = \
        ds.train_valid_test_split_randomly(
            num_valid=settings['num_valid'], 
            num_test=settings['num_test'], 
            augmentation=settings['augmentation'],
            shuffle=False, 
            random_seed=settings['random_seed'],
            task='real')
    print(X_train_rt.shape, X_valid_rt.shape, X_test_rt.shape, 
          Y_train_rt.shape, Y_valid_rt.shape, Y_test_rt.shape)

    dp_ds = Ellipsometry(data_type="with_loss", 
                         materials=real_materials)
    # settings['dataset'] = dp_ds
    X_train_dp, X_valid_dp, X_test_dp, Y_train_dp, Y_valid_dp, Y_test_dp = \
        dp_ds.train_valid_test_split_randomly(
            num_valid=settings['num_valid'], 
            num_test=settings['num_test'], 
            augmentation=settings['augmentation'],
            shuffle=False, 
            random_seed=settings['random_seed'],
            task='real')

    X_train = np.concatenate((X_train_dp, X_train_rt), axis=1)
    X_valid = np.concatenate((X_valid_dp, X_valid_rt), axis=1)
    X_test = np.concatenate((X_test_dp, X_test_rt), axis=1)
    Y_train = np.concatenate((Y_train_dp, Y_train_rt), axis=1)
    Y_valid = np.concatenate((Y_valid_dp, Y_valid_rt), axis=1)
    Y_test = np.concatenate((Y_test_dp, Y_test_rt), axis=1)

    dp_ds = Ellipsometry(data_type="with_loss", materials=real_materials)
    # settings['dataset'] = dp_ds
    _, _, _, Y_train_side, Y_valid_side, Y_test_side = \
        dp_ds.train_valid_test_split_randomly(
            num_valid=settings['num_valid'], 
            num_test=settings['num_test'], 
            augmentation=settings['augmentation'],
            shuffle=False, 
            random_seed=settings['random_seed'],
            task='inverse')

    extra_train = Y_train_side[0:2,6:10,:]
    extra_valid = Y_valid_side[0:1,6:10,:]
    extra_test = Y_test_side[0:1,6:10,:]

    phi_train = np.ones((2,1,256), dtype=float)*50.0/180*np.pi
    phi_valid = np.ones((1,1,256), dtype=float)*50.0/180*np.pi
    phi_test = np.ones((1,1,256), dtype=float)*50.0/180*np.pi
    d_train = np.ones((2,1,256), dtype=float)*0.3
    d_valid = np.ones((1,1,256), dtype=float)*0.3
    d_test = np.ones((1,1,256), dtype=float)*0.3
    Y_train = np.concatenate((Y_train, phi_train, d_train, extra_train), axis=1)
    Y_valid = np.concatenate((Y_valid, phi_valid, d_valid, extra_valid), axis=1)
    Y_test = np.concatenate((Y_test, phi_test, d_test, extra_test), axis=1)
    
    print(X_train.shape, X_valid.shape, X_test.shape, Y_train.shape, 
          Y_valid.shape, Y_test.shape)
    
    train_data = [X_train, X_valid, X_test, Y_train, Y_valid, Y_test]

    if args.train:
        pntrainer = PistonNetworkTrainer(settings=settings)
        pntrainer.train([X_train, X_train, X_train, Y_train, Y_train, Y_train])

    if args.test:
        device = torch.device("cuda")
        num_plots = 1
        X_data = np.concatenate([X_train, X_valid, X_test], axis=0)
        Y_data = np.concatenate([Y_train, Y_valid, Y_test], axis=0)
        X_data = X_data[0:num_plots,:,:]
        Y_data = Y_data[0:num_plots,:,:]

        net = torch.load(os.path.join(settings['models_dir'], 
            'best_single_net.tar'))
        net = net['net']
        net.eval()
        net = net.to(device).float()        

        experiment_dir = 'Results/Experiments/'+elem_folder
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        


#------------------------------------------------------------------------------#
if __name__ == "__main__":

    logger.info(settings)
    set_config(settings)
    main()    

    
