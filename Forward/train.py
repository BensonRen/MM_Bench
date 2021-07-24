"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import random
import os
import shutil
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import Forward
from utils.helper_functions import put_param_into_folder, write_flags_and_BVE

def training_from_flag(flags):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(Forward, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)


def retrain_different_dataset(index):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     from utils.helper_functions import load_flags
     data_set_list = ["Peurifoy"]
     for eval_model in data_set_list:
        model_dir = '_'.join([eval_model,'best_model'])
        flags = flag_reader.read_flag()
        flags.data_set = eval_model
        flags.model_name = "retrain" + str(index) + eval_model
        flags.geoboundary = [-1, 1, -1, 1]     # the geometry boundary of meta-material dataset is already normalized in current version
        flags.train_step = 300
        flags.batch_size = 1024
        flags.test_ratio = 0.2

        if 'Chen' in eval_model:
            flags.reg_scale = 0
            flags.lr_decay_rate = 0.4
            flags.linear = [201,1000,1000,1000,1000,1000,3]
            flags.linear = [256,700,700,700,700,700,700,700,700,5]
            flags.conv_kernel_size = []
            flags.conv_stride = []

        elif 'Peurifoy' in eval_model:
            flags.reg_scale = 1e-4
            flags.lr = 1e-4
            flags.lr_decay_rate = 0.6
            flags.linear = [201] + 15*[1700] + [8]
            flags.conv_kernel_size = []
            flags.conv_stride = []

        elif 'Yang' in eval_model:
            flags.reg_scale = 0
            flags.lr_decay_rate = 0.4
            flags.linear = [201,1000,1000,1000,1000,1000,3]
            flags.linear = [1990,1000,500,14]
            flags.conv_kernel_size = [7,5]
            flags.conv_stride = [1,1]

        print(flags)
        training_from_flag(flags)

def random_grid_search(dataset,rep=1):
    """
    This is for doing hyperswiping for the model parameters
    """

    for set in dataset:
        # Setup dataset folder
        spec_dir = os.path.join('models', set)

        if not os.path.exists(spec_dir):
            os.mkdir(spec_dir)

        dirs = os.listdir(spec_dir)

        # Clean up unfinished runs
        for run in dirs:
            d = os.path.join(spec_dir,run)
            for f in os.listdir(d):
                if f.find("training time.txt") != -1:
                    break
            else:
                shutil.rmtree(d)

        stride_vals = None
        kernel_vals = None

        if 'Chen' in set:
            reg_scale_list = [1e-5]  # [1e-4, 1e-3, 1e-2, 1e-1]
            layer_num = [13, 5, 8, 11]
            layer_size_list = [300, 1500, 700, 1100, 1900]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.0001,0.001]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.1,0.3]  # [0.1,0.3,0.5,0.7,0.9]
            ends = (256,5)

        elif 'Peurifoy' in set:
            reg_scale_list = [1e-3,1e-4,1e-5,0]  # [1e-4, 1e-3, 1e-2, 1e-1]
            layer_num = [10,13,15]
            layer_size_list = [1500,1700,2000]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.1,1e-2,1e-4]
            lr_decay = [0.1,0.2,0.3]
            ends = (201,8)

        elif 'Yang' in set:
            reg_scale_list = [1e-3,1e-4,1e-5,0]  # [1e-4, 1e-3, 1e-2, 1e-1]
            stride_vals = [1,2,3,4,5,6,7,8,9,10]
            kernel_vals = [2,3,4,5,6,7,8,9,10]
            layer_num = [17, 13, 5, 8, 11]
            layer_size_list = [300, 1500, 700, 1100, 1900]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.1,0.01,0.001,0.0001,1e-5]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.1,0.3,0.5,0.7]
            ends = (2000,14)
        else:
            return 0

        stride = []
        kernel = []

        while(True):
            ln = random.choice(layer_num)
            ls = random.choice(layer_size_list)
            lr = random.choice(lrate)
            reg_scale = random.choice(reg_scale_list)
            ld = random.choice(lr_decay)
            lin0 = ends[0]

            conv_config = ''
            if stride_vals and kernel_vals:
                num_convs = random.randrange(4)
                stride = []
                kernel = []

                if num_convs > 0:
                    for el in range(num_convs):
                        stride.append(random.choice(stride_vals))
                        kernel.append(random.choice(kernel_vals))

                    for s, k in iter(zip(stride, kernel)):
                        lin0 = 1 + (lin0 - k) / s

                    if not lin0.is_integer():
                        continue
                    else:
                        lin0 = int(lin0)
                    mat = ['kernel']+list(map(str,kernel))+['stride']+list(map(str,stride))
                    print(mat)
                    conv_config ='-'.join(mat)

            for i in range(rep):
                # If this combination has been tested before, name appropriately
                hyp_config = '_'.join(map(str, (ln, ls, lr, reg_scale, ld, conv_config)))  # Name by hyperparameters

                num_configs = 0                     # Count number of test instances
                for configs in dirs:
                    if hyp_config in configs:
                        num_configs += 1

                if num_configs >= rep:              # If # instances >= reps, make extra reps required or skip
                    continue

                name = '_'.join((hyp_config,str(num_configs)))

                # Model run
                flags = flag_reader.read_flag()
                flags.data_set = set                               # Save info
                flags.model_name = os.path.join(set,name)

                flags.linear = [ls for j in range(ln)]              # Architecture
                flags.linear[-1] = ends[-1]
                flags.conv_stride = stride
                flags.conv_kernel_size = kernel
                flags.linear[0] = lin0

                flags.lr = lr                                       # Other params
                flags.lr_decay_rate = ld
                flags.reg_scale = reg_scale
                flags.batch_size = 1024
                flags.train_step = 300
                flags.normalize_input = True

                training_from_flag(flags)

                dirs = os.listdir(spec_dir)


def hyperswipe(dataset, rep=1):
    """
    This is for doing hyperswiping for the model parameters
    """

    for set in dataset:
        # Setup dataset folder
        spec_dir = os.path.join('models', set)

        if not os.path.exists(spec_dir):
            os.mkdir(spec_dir)

        dirs = os.listdir(spec_dir)

        # Clean up unfinished runs
        for run in dirs:
            d = os.path.join(spec_dir,run)
            for f in os.listdir(d):
                if f.find("training time.txt") != -1:
                    break
            else:
                shutil.rmtree(d)

        stride = []
        kernel = []

        if 'Chen' in set:

            # Faster drops helped, but did add instability. lr 0.1 was too large, reg 1e-5 reduced instability but not enough
            # reg 1e-4 combined with faster decay = 0.3 and lower starting lr = 0.001 has great, consistent results
            # ^ reg = 0 -> High instability but results similar, reg = 1e-5 -> instability is lower
            # reg 1e-5 shown to be better than 0 or 1e-4, can be improved if lr gets lower later though lr=e-4,lr_decay=.1

            reg_scale_list = [1e-5]  # [1e-4, 1e-3, 1e-2, 1e-1]
            layer_num = [13, 5, 8, 11]
            layer_size_list = [300, 1500, 700, 1100, 1900]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.0001,0.001,0.01]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.1,0.3,0.4]  # [0.1,0.3,0.5,0.7,0.9]
            ends = (256,5)

        elif 'Peurifoy' in set:
            reg_scale_list = [0]  # [1e-4, 1e-3, 1e-2, 1e-1]
            layer_num = [7]
            layer_size_list = [300]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.1]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.4]  # [0.1,0.3,0.5,0.7,0.9]
            ends = (201,3)

        elif 'Yang' in set:
            reg_scale_list = [1e-4]  # [1e-4, 1e-3, 1e-2, 1e-1]
            stride = 0
            kernel = 0
            layer_num = [13, 5, 8, 11]
            layer_size_list = [300, 1500, 700, 1100, 1900]  # [1900,1500,1100,900,700,500,300]
            lrate = [0.001]  # [1e-1,1e-2,1e-4]
            lr_decay = [0.3]  # [0.1,0.3,0.5,0.7,0.9]
            ends = (2000,14)

        else:
            return 0

        for reg_scale in reg_scale_list:
            for ln in layer_num:
                for ls in layer_size_list:
                    for lr in lrate:
                        for ld in lr_decay:
                            for i in range(rep):
                                # If this combination has been tested before, name appropriately
                                hyp_config = '_'.join(map(str, (ln, ls, lr, reg_scale, ld)))  # Name by hyperparameters

                                num_configs = 0                     # Count number of test instances
                                for configs in dirs:
                                    if hyp_config in configs:
                                        num_configs += 1

                                if num_configs >= rep:              # If # instances >= reps, make extra reps required or skip
                                    continue

                                name = '_'.join((hyp_config,str(num_configs)))

                                # Model run
                                flags = flag_reader.read_flag()
                                flags.data_set = set                               # Save info
                                flags.model_name = os.path.join(set,name)

                                flags.linear = [ls for j in range(ln)]              # Architecture
                                flags.linear[0] = ends[0]
                                flags.linear[-1] = ends[-1]
                                flags.conv_stride = stride
                                flags.conv_kernel_size = kernel

                                flags.lr = lr                                       # Other params
                                flags.lr_decay_rate = ld
                                flags.reg_scale = reg_scale
                                flags.batch_size = 1024
                                flags.train_step = 300
                                flags.normalize_input = True

                                training_from_flag(flags)

                                dirs = os.listdir(spec_dir)         # Update dirs to include latest run


if __name__ == '__main__':
    # Read the parameters to be set
    #datasets = ['Chen']
    #hyperswipe(datasets, rep=2)

    datasets = ['Peurifoy']
    random_grid_search(datasets,rep=2)

    #for i in range(1):
    #    retrain_different_dataset(i)
