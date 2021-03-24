"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import NA
from utils import data_reader
from utils.helper_functions import load_flags
from utils.evaluation_helper import plotMSELossDistrib
from utils.evaluation_helper import get_test_ratio_helper
# Libs
import numpy as np
import matplotlib.pyplot as plt
from thop import profile, clever_format


def evaluate_from_model(model_dir, multi_flag=False, eval_data_all=False, save_misc=False, MSE_Simulator=False, save_Simulator_Ypred=True, init_lr=0.5, BDY_strength=1):

    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param multi_flag: The switch to turn on if you want to generate all different inference trial results
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
    :return: None
    """
    # Retrieve the flag object
    print("Retrieving flag object for parameters")
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    print(model_dir)
    flags = load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode
    flags.backprop_step = eval_flags.backprop_step
    flags.test_ratio = get_test_ratio_helper(flags)

    if flags.data_set == 'Yang_sim':
        save_Simulator_Ypred = False
        print("this is Yang sim dataset, setting the save_Simulator_Ypred to False")
    flags.batch_size = 1                            # For backprop eval mode, batchsize is always 1
    flags.lr = init_lr
    flags.BDY_strength = BDY_strength
    flags.eval_batch_size = eval_flags.eval_batch_size
    flags.train_step = eval_flags.train_step

    # delete after usage: 02.07 for vilidating that ball and sine is convex problem
    # Use a very small eval batch size and expected to see that meta and robo getting much worse performance
    # and the ball and sine getting nearly identical one
    # flags.eval_batch_size = 2

    print(flags)

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, eval_data_all=eval_data_all)
    print("Making network now")
    
    # Make Network
    ntwk = Network(NA, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    

    # Evaluation process
    print("Start eval now:")
    if multi_flag:
        dest_dir = '/home/sr365/MM_bench_multi_eval/NA/'
        #dest_dir = '/data/users/ben/multi_eval/NA_lr' + str(init_lr)  + 'bdy_' + str(BDY_strength)+'/' 
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        dest_dir += flags.data_set
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        #pred_file, truth_file = ntwk.evaluate(save_dir='/work/sr365/multi_eval/NA/' + flags.data_set, save_all=True,
        pred_file, truth_file = ntwk.evaluate(save_dir=dest_dir, save_all=True,
                                                save_misc=save_misc, MSE_Simulator=MSE_Simulator,save_Simulator_Ypred=save_Simulator_Ypred)
    else:
        pred_file, truth_file = ntwk.evaluate(save_misc=save_misc, MSE_Simulator=MSE_Simulator, save_Simulator_Ypred=save_Simulator_Ypred)

    # Plot the MSE distribution
    plotMSELossDistrib(pred_file, truth_file, flags)
    print("Evaluation finished")


def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None

def evaluate_different_dataset(multi_flag, eval_data_all, save_Simulator_Ypred=False, MSE_Simulator=False):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     ## Evaluate all models with "reatrain" and dataset name in models/
     for model in os.listdir('models/'):
         if 'best' in model:
             evaluate_from_model(model, multi_flag=multi_flag, 
                          eval_data_all=eval_data_all,save_Simulator_Ypred=save_Simulator_Ypred, MSE_Simulator=MSE_Simulator)

def evaluate_trail_BDY_lr(multi_flag, eval_data_all, save_Simulator_Ypred=False, MSE_Simulator=False):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     #lr_list = [2, 1,0.5,0.1]
     lr_list = [0.5]
     BDY_list = [0.001]
     #BDY_list = [0.05, 0.01, 0.001]
     #data_set_list = ["robotic_arm"]
     #data_set_list = ["robotic_arm", "ballistics"]
     for eval_model in data_set_list:
        for lr in lr_list:
            for BDY in BDY_list:
                useless_flags = flag_reader.read_flag()
                useless_flags.eval_model = "retrain5" + eval_model
                evaluate_from_model(useless_flags.eval_model, multi_flag=multi_flag, eval_data_all=eval_data_all, save_Simulator_Ypred=save_Simulator_Ypred, MSE_Simulator=MSE_Simulator, init_lr = lr, BDY_strength=BDY)

if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    eval_flags = flag_reader.read_flag()
    
    #####################
    # different dataset #
    #####################
    # This is to run the single evaluation, please run this first to make sure the current model is well-trained before going to the multiple evaluation code below
    evaluate_different_dataset(multi_flag=False, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)
    # This is for multi evaluation for generating the Fig 3, evaluating the models under various T values
    #evaluate_different_dataset(multi_flag=True, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)
    
    # This is to test the BDY and LR effect of the NA method specially for Robo and Ballistics dataset, 2021.01.09 code trail for investigating why sometimes NA constrait the other methods
    #evaluate_trail_BDY_lr(multi_flag=True, eval_data_all=False, save_Simulator_Ypred=True, MSE_Simulator=False)
