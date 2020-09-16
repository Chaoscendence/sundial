import os
from os.path import join
import pandas as pd
import numpy as np
import torch
from numpy.polynomial import polynomial as P
from numpy import linalg
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as utils_shuffle
from sundial.utils import set_my_favorite_plot_settings
from sundial.training_piston_network import forward_program
import matplotlib.pyplot as plt
set_my_favorite_plot_settings()
from meta_settings import *

DATASET_DIR = "sundial/datasets/"

def set_with_order(x_list):
    x_set = []
    for x in x_list:
        if x not in x_set:
            x_set.append(x) 
    return x_set

FileNamePostfix = '_'+'L'+str(MS_LAMBDA_LOW_BOUND)+'U'+str(MS_LAMBDA_UP_BOUND)

class Ellipsometry(object):
    def __init__(self, data_type='with_loss', materials='all'):
        assert data_type.lower() in ['with_loss']
        if data_type.lower() == 'with_loss':
            self.data_dir = join(DATASET_DIR, 'with_materials_loss')
        print("Ellipsometry: " + FileNamePostfix)

        self.inverse_inputs = np.load(os.path.join(self.data_dir,
                                'inverse_inputs'+FileNamePostfix+'.npy'))
        self.inverse_targets = np.load(os.path.join(self.data_dir,
                                'inverse_targets'+FileNamePostfix+'.npy'))
        self.forward_inputs = np.load(os.path.join(self.data_dir,
                                'forward_inputs'+FileNamePostfix+'.npy'))
        self.forward_targets = np.load(os.path.join(self.data_dir,
                                'forward_targets'+FileNamePostfix+'.npy'))
        self.real_inputs = np.load(os.path.join(self.data_dir,
                                'test_inputs'+FileNamePostfix+'.npy'))
        self.real_targets = np.load(os.path.join(self.data_dir,
                                'test_targets'+FileNamePostfix+'.npy'))
        self.real_metas = torch.load(os.path.join(self.data_dir,
                                'test_metas.tar'))
        real_index = 0 
        for x in self.real_metas:
            if x.lower() == materials.lower():
                break
            real_index += 1
        if (real_index >= len(self.real_metas)) and (materials.lower() != 'all'):
            print("---Materials {} NOT found---".format(materials))
            exit()
        elif materials.lower() != 'all':
            print("---Data for Materials {} prepared---".format(materials))
            self.real_inputs = self.real_inputs[real_index:real_index+1,:,:]
            self.real_targets = self.real_targets[real_index:real_index+1,:,:]            
        else:            
            print("---Data for Materials {} prepared---".format(materials))
        
        self.materials = materials
 
    def train_valid_test_split_randomly(self, num_valid, 
                                        num_test, 
                                        augmentation=False, 
                                        shuffle=True, 
                                        random_seed=0, 
                                        task='inverse'):
        if task.lower() == 'inverse':
            self.inputs = self.inverse_inputs
            self.targets = self.inverse_targets
        elif task.lower() == 'forward':
            self.inputs = self.forward_inputs
            self.targets = self.forward_targets
        elif task.lower() == 'real':
            inputs = [self.real_inputs]*4
            targets = [self.real_targets]*4
            self.inputs = np.concatenate(inputs, axis=0)
            self.targets = np.concatenate(targets, axis=0)
        else:
            print("Please specify a correct task, inverse, forward or real")
        num_samples, num_input_channels, len_x = self.inputs.shape
        num_samples, num_target_channels, len_y = self.targets.shape
        num_train = num_samples - num_valid - num_test
        X = pd.DataFrame(self.inputs.reshape(num_samples,-1))
        Y = pd.DataFrame(self.targets.reshape(num_samples,-1))
        X_left, X_test, Y_left, Y_test = train_test_split(X, Y, 
            test_size=num_test, random_state=random_seed, shuffle=shuffle)
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_left, Y_left,
            test_size=num_valid, random_state=random_seed, shuffle=shuffle)

        self.testset_index = Y_test.index.values

        X_train = X_train.values.reshape(num_train, num_input_channels, len_x)
        X_valid = X_valid.values.reshape(num_valid, num_input_channels, len_x)
        X_test = X_test.values.reshape(num_test, num_input_channels, len_x)
        Y_train = Y_train.values.reshape(num_train, num_target_channels, len_y)
        Y_valid = Y_valid.values.reshape(num_valid, num_target_channels, len_y)
        Y_test = Y_test.values.reshape(num_test, num_target_channels, len_y)
       
        if augmentation:
            X_aug, Y_aug = self.random_augmentation(X_train, Y_train, 
                                                    num_shift_x_per_sample=5,
                                                    max_shift_x=5, 
                                                    num_shift_y_per_sample=5,
                                                    max_shift_y=1,
                                                    num_mag_per_sample=5,
                                                    max_mag_noise=0.001)
            X_train = np.concatenate((X_train, X_aug), axis=0)
            Y_train = np.concatenate((Y_train, Y_aug), axis=0)
        
        if shuffle:
            X_train, Y_train = utils_shuffle(X_train, Y_train, 
                                             random_state=random_seed)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def train_valid_test_split_by_categories(self, augmentation=False,
                                             shuffle=True, 
                                             random_seed=0, 
                                             task='inverse'):
        if task.lower() == 'inverse':
            self.inputs = self.inverse_inputs
            self.targets = self.inverse_targets
        elif task.lower() == 'forward':
            self.inputs = self.forward_inputs
            self.targets = self.forward_targets
        elif task.lower() == 'real':
            self.inputs = self.real_inputs
            self.targets = self.real_inputs
        else:
            print("Please specify a correct task, inverse, forward or real")

        train_indexes = []
        valid_indexes = []
        test_indexes = []
        test_names_found = []
        valid_names_found = []
        train_names_found = []
        for i, item in enumerate(self.top_materials_names):
            if item in self.test_names:
                test_indexes.append(i)
                test_names_found.append(item)
                continue
            if item in self.valid_names:
                valid_indexes.append(i)
                valid_names_found.append(item)
                continue
            if item in self.train_names:
                train_indexes.append(i)
                train_names_found.append(item)

        train_indexes = np.array(train_indexes)
        valid_indexes = np.array(valid_indexes)
        test_indexes = np.array(test_indexes)
        # test_names_found = list(set(test_names_found))
        # valid_names_found = list(set(valid_names_found))
        # test_names_found.sort()
        # valid_names_found.sort()
        self.train_names = dict(zip(set_with_order(train_names_found), 
                                    range(len(train_names_found)))) 
        self.valid_names = dict(zip(set_with_order(valid_names_found), 
                                    range(len(valid_names_found)))) 
        self.test_names = dict(zip(set_with_order(test_names_found), 
                                    range(len(test_names_found)))) 
        X_train = self.inputs[train_indexes,:,:]
        X_valid = self.inputs[valid_indexes,:,:]
        X_test = self.inputs[test_indexes,:,:]
        Y_train = self.targets[train_indexes,:,:]
        Y_valid = self.targets[valid_indexes,:,:]
        Y_test = self.targets[test_indexes,:,:]

        if augmentation:
            X_aug, Y_aug = self.random_augmentation(X_train, Y_train, 
                                                    num_shift_x_per_sample=10,
                                                    max_shift_x=5, 
                                                    num_shift_y_per_sample=10,
                                                    max_shift_y=1,
                                                    num_mag_per_sample=10,
                                                    max_mag_noise=0.001)
            X_train = np.concatenate((X_train, X_aug), axis=0)
            Y_train = np.concatenate((Y_train, Y_aug), axis=0)

        if False:
            delta_psi = X_train[:,0:2,:]
            phi1_d = X_train[:,2:4,:]
            rest = X_train[:,4:,:]
            X = np.concatenate([Y_train, phi1_d, rest], axis=1)
            Y = delta_psi
            X_aug_nb, Y_aug_nb = self.neighborhood(X, Y, 
                                                   num_shift_per_sample=5,
                                                   max_wave_shift=5, 
                                                   num_mag_per_sample=5,
                                                   max_mag_noise=0.001,
                                                   num_direction_per_sample=10)
            X_aug = np.concatenate((Y_aug_nb, X_aug_nb[:,4:6,:], 
                                    X_aug_nb[:,6:,:]), axis=1)
            Y_aug = X_aug_nb[:,0:4,:]
            X_train = np.concatenate((X_train, X_aug), axis=0)
            Y_train = np.concatenate((Y_train, Y_aug), axis=0)

        if shuffle:
            X_train, Y_train = utils_shuffle(X_train, Y_train, 
                                             random_state=random_seed)

        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def random_augmentation(self, X, Y, 
                            num_shift_x_per_sample, 
                            max_shift_x, 
                            num_shift_y_per_sample, 
                            max_shift_y=1.0,
                            num_mag_per_sample=0, 
                            max_mag_noise=0.001,
            #                   num_pow_per_sample, 
                            ):
        """
        machine learning way of data augmentation
        """                        
        num_samples, n_channels, features = X.shape        

        # Shift the spectra horizontally/along x axis
        X_aug = np.array([]).reshape(0, X.shape[1], X.shape[2])
        Y_aug = np.array([]).reshape(0, Y.shape[1], Y.shape[2])
        for i in range(num_shift_x_per_sample):
            X_aug_shift = np.copy(X)
            Y_aug_shift = np.copy(Y)
            n_right = np.random.randint(-max_shift_x, max_shift_x)
            X_aug_shift = np.roll(X_aug_shift, n_right, axis=2)
            X_aug = np.concatenate((X_aug, X_aug_shift), axis=0)
            Y_aug = np.concatenate((Y_aug, Y_aug_shift), axis=0)

        # Shift the spectra vertically/along y axis
        # Note that here a whole channel of spectra is shifted up or down by 
        # a random number
        for i in range(num_shift_y_per_sample):
            X_aug_shift_y = np.copy(X)
            Y_aug_shift_y = np.copy(Y)
            s1, s2, s3 = X_aug_shift_y.shape
            rs = np.random.randn(s1, s2)*max_shift_y
            mu = np.mean(X_aug_shift_y, axis=2)
            noise_y = np.tile(rs*mu, (s3, 1, 1)).transpose((1, 2, 0))
            X_aug_shift_y = X_aug_shift_y + noise_y
            X_aug = np.concatenate((X_aug, X_aug_shift_y), axis=0)
            Y_aug = np.concatenate((Y_aug, Y_aug_shift_y), axis=0)

        # Add random noises along y axis
        # Note that here each wavenumber is shifted invidually up or down. 
        for i in range(num_mag_per_sample):
            X_aug_mag = np.copy(X)         
            s1, s2, s3 = X_aug_mag.shape
            random_scaling = np.random.randn(s1, s2, s3)
            X_aug_mag = np.multiply(X_aug_mag, 1+random_scaling*max_mag_noise)
            Y_aug_mag = np.copy(Y)            
            X_aug = np.concatenate((X_aug, X_aug_mag), axis=0)
            Y_aug = np.concatenate((Y_aug, Y_aug_mag), axis=0)

        # Apply power functions to X. 
        # Seem this is not very useful for ellipsometry data. Uncomment these
        # lines to enable power-style augmentation.        
        # for i in range(num_pow_per_sample-1):
        #     p = 5.0 * (i + 1.0) / num_pow_per_sample
        #     X_aug_power = np.copy(X)
        #     X_aug_power = np.power(X_aug_power, p)
        #     Y_aug_power = np.copy(Y)            
        #     X_aug = np.vstack([X_aug, X_aug_power])
        #     Y_aug = np.vstack([Y_aug, Y_aug_power])

        return X_aug, Y_aug

    def neighborhood(self, X, Y, 
                     num_shift_per_sample, 
                     max_wave_shift, 
                     num_mag_per_sample, 
                     max_mag_noise,
                     num_direction_per_sample=0,
                     nkz_only=False,
                     combine=True):
        """
        Generate samples in the neighborhood of a give pair of samples (X, Y)
        
        num_shift_per_sample:
            Shift the whole spectrum left or right randomly

        num_mag_per_sample:
            Add random noise along y-axis

        num_direction_per_sample:        
            Generating data along random directions. So hopefully after training 
            on this set of data, the network can approximate the objective
            function along many random directions so that the gradient of the 
            network can approximate that of the objective function well.

        nkz_only: if True, only augment nkz

        combine: if True, return both augmented and the original data
        """
        num_samples, n_channels, features = X.shape
        X_copy = X
        Y_copy = Y

        # Shift
        X_aug = np.array([]).reshape(0, X_copy.shape[1], X_copy.shape[2])
        Y_aug = np.array([]).reshape(0, Y_copy.shape[1], Y_copy.shape[2])
        for i in range(num_shift_per_sample):
            X_aug_shift = np.copy(X_copy)
            n_right = np.random.randint(-max_wave_shift, max_wave_shift)
            X_aug_shift = np.roll(X_aug_shift, n_right, axis=2)
            for i in range(n_right):
                X_aug_shift[:,:,i] = X_aug_shift[:,:,n_right+1]
            Y_aug_shift = forward_program(X_aug_shift)
            X_aug = np.concatenate((X_aug, X_aug_shift), axis=0)
            Y_aug = np.concatenate((Y_aug, Y_aug_shift), axis=0)
        
        # Add random noises along y axis
        for i in range(0):
            X_aug_mag = np.copy(X_copy)   
            s1, s2, s3 = X_aug_mag.shape
            random_scaling = np.random.randn(s1, s2, s3)
            X_aug_mag = np.multiply(X_aug_mag, 1+random_scaling*max_mag_noise)
            Y_aug_mag = forward_program(X_aug_mag)            
            X_aug = np.concatenate((X_aug, X_aug_mag), axis=0)
            Y_aug = np.concatenate((Y_aug, Y_aug_mag), axis=0)

        for i in range(num_mag_per_sample):
            X_aug_mag = np.copy(X_copy)   
            # random_scaling = np.random.rand()*0.2+0.9
            # X_aug_mag = np.multiply(X_aug_mag, random_scaling)
            random_scaling = np.random.rand()
            X_aug_mag = np.multiply(X_aug_mag, 1+random_scaling*max_mag_noise)
            Y_aug_mag = forward_program(X_aug_mag)
            X_aug = np.concatenate((X_aug, X_aug_mag), axis=0)
            Y_aug = np.concatenate((Y_aug, Y_aug_mag), axis=0)

        steps = [np.power(10,-float(x)) for x in range(3,6)]
        for i in range(num_direction_per_sample):
            for s in steps:
                X_aug_mag = np.copy(X_copy)   
                s1, s2, s3 = X_aug_mag.shape
                directions = np.random.randn(s1, s2*s3)
                l2norm = np.sqrt(np.sum(np.power(directions, 2), axis=1))
                l2norm = np.matmul(l2norm.reshape((s1,1)), 
                    np.ones((1,s2*s3), dtype=float))
                directions = np.reshape(directions/l2norm, (s1,s2,s3))
                X_aug_mag += s*directions
                # delta, psi = compute_delta_psi_from_nkz(nkz=X_aug_mag, phi=phi, 
                #                                         d=d, combine=False)
                # Y_aug_mag = np.transpose(np.array([delta, psi]), (1,0,2))
                Y_aug_mag = forward_program(X_aug_mag)
                X_aug = np.concatenate((X_aug, X_aug_mag), axis=0)
                Y_aug = np.concatenate((Y_aug, Y_aug_mag), axis=0)

        # Directions along axis
        # steps = [1e-5, 5e-4, 1e-4, 5e-3, 1e-3]
        # steps = np.random.rand(10)*(1e-3 - 1e-5) + 1e-5
        # for i in range(10):
        #     s = steps[i]
        #     X_aug_mag_c = np.copy(X_copy)   
        #     s1, s2, s3 = X_aug_mag_c.shape
        #     directions = np.eye(s2*s3).reshape(s2*s3,s1,s2,s3)
        #     X_aug_mag = np.tile(X_aug_mag_c, (s2*s3,1,1,1))
        #     X_aug_mag = (X_aug_mag + s*directions).reshape(-1,s2,s3)
        #     delta, psi = compute_delta_psi_from_nkz(X_aug_mag, combine=False)
        #     Y_aug_mag = np.transpose(np.array([delta, psi]), (1,0,2))
        #     X_aug = np.concatenate((X_aug, X_aug_mag), axis=0)
        #     Y_aug = np.concatenate((Y_aug, Y_aug_mag), axis=0)

        if combine:
            X_aug = np.concatenate((X, X_aug), axis=0) 
            Y_aug = np.concatenate((Y, Y_aug), axis=0) 

        return X_aug, Y_aug

    def affine_transform(self, X, Y):
        """ Apply affine transform to X
        """
        X_aug = np.array([]).reshape(0, X.shape[1], X.shape[2])
        Y_aug = np.array([]).reshape(0, Y.shape[1], Y.shape[2])

        # Shift the spectra vertically/along y axis
        X_c = np.copy(X)
        Y_c = np.copy(Y)
        s1, s2, s3 = X_c.shape
        rs = np.random.randn(s1, s2)
        mu = np.mean(X_c, axis=2)
        noise_y = np.tile(rs*mu, (s3, 1, 1)).transpose((1, 2, 0))
        X_c = X_c + noise_y
        X_aug = np.concatenate((X_aug, X_c), axis=0)
        Y_aug = np.concatenate((Y_aug, Y_c), axis=0)

        # Rotate along the center 
        y = X[0,0,:]
        x = np.linspace(1, 2, num=256, endpoint=True)

        xy = np.array([x,y])
        mu = np.mean(xy, axis=1).reshape((2,1))
        mu = np.matmul(mu, np.ones((1,256), dtype=np.float))
        xy -= mu
        theta = 3.0/180*np.pi*0.1
        rot_mat = np.array([[np.cos(theta), np.sin(theta)],
                           [-np.sin(theta), np.cos(theta)]])
        xy_rot = np.matmul(rot_mat, xy) 
        xy_rot += mu
        xy += mu
        print(rot_mat)
        print(xy.shape, xy_rot.shape)
        plt.plot(xy[0,:], xy[1,:], 'b')
        plt.plot(xy_rot[0,:], xy_rot[1,:], 'g')
        plt.ylim([3.0, 3.2])
        plt.show()

        return X_aug, Y_aug
      


