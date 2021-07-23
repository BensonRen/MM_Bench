"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys
sys.path.append('../utils/')

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler

# Libs
import numpy as np
from math import inf
import matplotlib.pyplot as plt
import pandas as pd
# Own module
from utils.time_recorder import time_keeper
from utils.helper_functions import simulator

class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            if saved_model.startswith('models/'):
                saved_model = saved_model.replace('models/','')
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model) #saved_model
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:                    # leave custume name if possible
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.loss = self.make_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.optm_eval = None                                   # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)                 # Create a summary writer for keeping the summary to the tensor board
        if not os.path.isdir(self.ckpt_dir) and not inference_mode:
            os.mkdir(self.ckpt_dir)
        self.best_validation_loss = float('inf')                # Set the BVL to large number

    def make_optimizer_eval(self, geometry_eval, optimizer_type=None):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD([geometry_eval], lr=self.flags.lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_size=(128, 8))
        print(model)
        return model

    def make_loss(self, logit=None, labels=None, G=None, return_long=False):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :param larger_BDY_penalty: For only filtering experiments, a larger BDY penalty is added
        :param return_long: The flag to return a long list of loss in stead of a single loss value,
                            This is for the forward filtering part to consider the loss
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss
        # BDY_loss = 0
        # if G is not None:         # This is using the boundary loss
        #     X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
        #     X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
        #     relu = torch.nn.ReLU()
        #     BDY_loss_all = 1 * relu(torch.abs(G - self.build_tensor(X_mean)) - 0.5 * self.build_tensor(X_range))
        #     BDY_loss = 0.1*torch.sum(BDY_loss_all)
        #     #BDY_loss = self.flags.BDY_strength*torch.sum(BDY_loss_all)
        # self.MSE_loss = MSE_loss
        # self.Boundary_loss = BDY_loss
        # return torch.add(MSE_loss, BDY_loss)
        return MSE_loss


    def build_tensor(self, nparray, requires_grad=False):
        return torch.tensor(nparray, requires_grad=requires_grad, device='cuda', dtype=torch.float)


    def make_optimizer(self, optimizer_type=None):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        # For eval mode to change to other optimizers
        if  optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_lr_scheduler(self, optm):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        #torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'best_model.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        if torch.cuda.is_available():
            #self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model.pt')))
        else:
            #self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'), map_location=torch.device('cpu'))
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model.pt'), map_location=torch.device('cpu')))

    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total Number of Parameters: {}".format(pytorch_total_params))

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            # boundary_loss = 0                 # Unnecessary during training since we provide geometries
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                logit = self.model(spectra)                        # Get the output
                loss = self.make_loss(logit, geometry)               # Get the loss tensor
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss                                  # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/train', train_avg_loss, epoch)
                # self.log.add_scalar('Loss/BDY_train', boundary_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logit = self.model(spectra)
                    loss = self.make_loss(logit, geometry)                   # compute the loss
                    test_loss += loss                                       # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j+1)
                self.log.add_scalar('Loss/test', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        self.log.close()
        tk.record(1)                    # Record at the end of the training

    def evaluate(self, save_dir='data/', save_all=False, MSE_Simulator=False, save_misc=False, save_Simulator_Ypred=True):
        """
        The function to evaluate how good the models is (outputs validation loss)
        Note that Ypred and Ytruth still refer to spectra, while Xpred and Xtruth still refer to geometries.
        :return:
        """

        self.load()                             # load the model as constructed

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()
        saved_model_str = self.saved_model.replace('/','_')
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str)) #Input associated? No real value
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str)) #Output to compare against
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str)) #Input of Neural Net
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str)) #Output of Neural Net
        print("evalution output pattern:", Ypred_file)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(save_dir, 'evaluation_time.txt'))

        # Open those files to append
        with open(Xtruth_file, 'w') as fxt,open(Ytruth_file, 'w') as fyt,\
                open(Ypred_file, 'w') as fyp, open(Xpred_file, 'w') as fxp:

            # Loop through the eval data and evaluate
            geometry, spectra = next(iter(self.test_loader))

            if cuda:
                geometry = geometry.cuda()
                spectra = spectra.cuda()

            # Initialize the geometry first
            Xpred = self.model(spectra).cpu().data.numpy()
            Ytruth = spectra.cpu().data.numpy()

            if save_Simulator_Ypred and not(self.flags.data_set == 'Yang' or self.flags.data_set == 'Yang_sim'):
                Ypred = simulator(self.flags.data_set, Xpred)
            else:
                Ypred = spectra.cpu().data.numpy()

            MSE_List = np.mean(np.power(Ypred - Ytruth, 2), axis=1)
            mse = np.mean(MSE_List)
            print(mse)

            np.savetxt(fxt, geometry.cpu().data.numpy())
            np.savetxt(fyt, Ytruth)
            if self.flags.data_set != 'Yang':
                np.savetxt(fyp, Ypred)
            np.savetxt(fxp, Xpred)

        return Ypred_file, Ytruth_file


    def plot_histogram(self, loss, ind):
        """
        Plot the loss histogram to see the loss distribution
        """
        f = plt.figure()
        plt.hist(loss, bins=100)
        plt.xlabel('MSE loss')
        plt.ylabel('cnt')
        plt.suptitle('(Avg MSE={:4e})'.format(np.mean(loss)))
        plt.savefig(os.path.join('data','loss{}.png'.format(ind)))
        return None
