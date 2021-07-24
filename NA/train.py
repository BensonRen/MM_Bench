"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import NA
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
    ntwk = Network(NA, flags, train_loader, test_loader)

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
     #data_set_list = ["ballistics"]
     #data_set_list = ["meta_material"]
     data_set_list = ["meta_material","robotic_arm","sine_wave","ballistics"]
     for eval_model in data_set_list:
        flags = load_flags(os.path.join("models", eval_model))
        flags.model_name = "retrain" + str(index) + eval_model
        flags.geoboundary = [-1, 1, -1, 1]     # the geometry boundary of meta-material dataset is already normalized in current version
        flags.train_step = 500
        flags.test_ratio = 0.2
        training_from_flag(flags)

def hyperswipe():
    """
    This is for doing hyperswiping for the model parameters
    """
    layer_size_list = [5000, 1000]
    #layer_size_list = [1750]
    #layer_size_list = [100, 250, 500]
    reg_scale_list = [1e-4]
    #layer_num = 7
    for reg_scale in reg_scale_list:
        for i in range(1):
            for layer_num in range(8, 25, 2):
                for layer_size in layer_size_list:
                    flags = flag_reader.read_flag()  	#setting the base case
                    linear = [layer_size for j in range(layer_num)]        #Set the linear units
                    linear[0] = 10                   # The start of linear
                    linear[-1] = 1001                # The end of linear
                    flags.lr = 1e-4
                    flags.linear = linear
                    flags.reg_scale = reg_scale
                    #flags.conv_kernel_size = [3, 3, 5]
                    #flags.conv_channel_out = [4, 4, 4]
                    #flags.conv_stride = [1, 1, 1]
                    flags.model_name = flags.data_set + 'no_conv_' + str(layer_size) + '_num_' + str(layer_num) + '_lr_' + str(flags.lr) + 'reg_scale_' + str(reg_scale) + 'trail_' + str(i)
                    #flags.model_name = flags.data_set + 'conv_444_335_111_linear_' + str(layer_size) + '_num_' + str(layer_num) + '_lr_' + str(flags.lr) + 'reg_scale_' + str(reg_scale) + 'trail_' + str(i)
                    training_from_flag(flags)

if __name__ == '__main__':
    # Read the parameters to be set
    """
    for i in range(5):
        flags.model_name = 'Yang_param_pure_' + str(i+15)
        #linear = [1000  for j in range(11)]
        #linear[0] = 14
        #linear[-1] = 500
        #flags.linear = linear
        #flags.reg_scale = 1e-4
        training_from_flag(flags)
    """
    hyperswipe()
    #flags = flag_reader.read_flag()  	#setting the base case
    #training_from_flag(flags)
    # Do the retraining for all the data set to get the training 
    #for i in range(10):
    #    retrain_different_dataset(i)

