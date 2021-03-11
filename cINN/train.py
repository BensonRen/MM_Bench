"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil

# Torch
import torch
# Own
import flag_reader
from utils import data_reader
from class_wrapper import Network
from model_maker import cINN
from utils.helper_functions import put_param_into_folder,write_flags_and_BVE
from evaluate import evaluate_from_model

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
    ntwk = Network(cINN, flags, train_loader, test_loader)

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
    #reg_scale_list = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    data_set_list = ["robotic_arm","sine_wave","ballistics","meta_material"]
    for eval_model in data_set_list:
        #for reg_scale in reg_scale_list:
        flags = load_flags(os.path.join("models", eval_model))
        # 0124 trail
        #flags.model_name = "retrain" + str(index) + str(reg_scale) + eval_model
        #flags.reg_scale = reg_scale
        flags.model_name = "retrain" + str(index) + eval_model
        flags.batch_size = 1024
        flags.geoboundary = [-1, 1, -1, 1]     # the geometry boundary of meta-material dataset is already normalized in current version
        flags.train_step = 500
        flags.test_ratio = 0.2
        #flags.reg_scale = 0.08
        flags.stop_threshold = -float('inf')
        training_from_flag(flags)

def hyperswipe():
    """
    This is for doing hyperswiping for the model parameters
    """
    reg_scale_list = [5e-3]
    lambda_mse_list = [100, 10, 1]
    for reg_scale in reg_scale_list:
        for couple_layer_num in range(4,10):    
            for lambda_mse in lambda_mse_list:
                flags = flag_reader.read_flag()  	#setting the base case
                flags.couple_layer_num = couple_layer_num
                flags.lambda_mse = lambda_mse
                flags.reg_scale = reg_scale
                flags.model_name = flags.data_set + 'couple_layer_num' + str(couple_layer_num) + 'labmda_mse' + str(lambda_mse) + '_lr_' + str(flags.lr) + '_reg_scale_' + str(reg_scale)
                training_from_flag(flags)

if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()
    
    hyperswipe()
    # Call the train from flag function
    #training_from_flag(flags)

    # Do the retraining for all the data set to get the training for reproducibility
    #for i in range(5):
    #    retrain_different_dataset(i)
