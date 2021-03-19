import os
import time
import torch
import math
from utils.helper_functions import simulator
import numpy as np
from utils.time_recorder import time_keeper
from utils.evaluation_helper import plotMSELossDistrib
from model_maker import Net
import sys
from torch.utils.tensorboard import SummaryWriter

# TODO: Over training problems
# TODO: Have ga.py only hold GA object, and slip GA_manager functionality into class_wrapper

class GA_manager(object):
    def __init__(self, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None, GA_eval_mode=False):
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            if saved_model.startswith('models/'):
                saved_model = saved_model.replace('models/','')
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:                    # leave custume name if possible
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        if not os.path.isdir(self.ckpt_dir) and not inference_mode:
            os.mkdir(self.ckpt_dir)

        self.model = Net(flags)
        self.load()
        self.model.cuda().eval()
        self.algorithm = GA(flags, self.model, self.make_loss)

        self.best_validation_loss = float('inf')                # Set the BVL to large number

    #TODO: Fill in all necessary class_wrapper functions
    def make_loss(self, logit,labels):
        return torch.mean(torch.pow(logit - labels,2), dim=1)

    def load(self):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model.pt')))
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model.pt'), map_location=torch.device('cpu')))

    def evaluate(self, save_dir='data/', save_all=False, MSE_Simulator=False, save_misc=False, save_Simulator_Ypred=True):
        """
        The function to evaluate how good the Neural Adjoint is and output results
        :param save_dir: The directory to save the results
        :param save_all: Save all the results instead of the best one (T_200 is the top 200 ones)
        :param MSE_Simulator: Use simulator loss to sort (DO NOT ENABLE THIS, THIS IS OK ONLY IF YOUR APPLICATION IS FAST VERIFYING)
        :param save_misc: save all the details that are probably useless
        :param save_Simulator_Ypred: Save the Ypred that the Simulator gives
        (This is useful as it gives us the true Ypred instead of the Ypred that the network "thinks" it gets, which is
        usually inaccurate due to forward model error)
        :return:
        """
        try:
            bs = self.flags.generations         # for previous code that did not incorporate this
        except AttributeError:
            print("There is no attribute backprop_step, catched error and adding this now")
            self.flags.generations = 300
        cuda = True if torch.cuda.is_available() else False
        #if cuda:
        #    self.model.cuda()
        #self.model.eval()
        saved_model_str = self.saved_model.replace('/','_')
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))
        print("evalution output pattern:", Ypred_file)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(save_dir, 'evaluation_time.txt'))

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                # Initialize the geometry first
                Xpred, Ypred, loss = self.evaluate_one(spectra, save_dir=save_dir, save_all=save_all, ind=ind,
                                                        MSE_Simulator=MSE_Simulator, save_misc=save_misc, save_Simulator_Ypred=save_Simulator_Ypred)
                tk.record(ind)                          # Keep the time after each evaluation for backprop
                # self.plot_histogram(loss, ind)                                # Debugging purposes
                np.savetxt(fxt, geometry.cpu().data.numpy())
                np.savetxt(fyt, spectra.cpu().data.numpy())
                if self.flags.data_set != 'Yang':
                    np.savetxt(fyp, Ypred)
                np.savetxt(fxp, Xpred)
                plotMSELossDistrib(Ypred_file,Ytruth_file,self.flags)
        return Ypred_file, Ytruth_file

    def evaluate_one(self, target_spectra, save_dir='data/', MSE_Simulator=False, save_all=False, ind=None,
                     save_misc=False, save_Simulator_Ypred=True, init_from_Xpred=None, FF=True):
        """
        The function which being called during evaluation and evaluates one target y using # different trails
        :param target_spectra: The target spectra/y to backprop to
        :param save_dir: The directory to save to when save_all flag is true
        :param MSE_Simulator: Use Simulator Loss to get the best instead of the default NN output logit
        :param save_all: The multi_evaluation where each trail is monitored (instad of the best) during backpropagation
        :param ind: The index of this target_spectra in the batch
        :param save_misc: The flag to print misc information for degbugging purposes, usually printed to best_mse
        :return: Xpred_best: The 1 single best Xpred corresponds to the best Ypred that is being backproped
        :return: Ypred_best: The 1 singe best Ypred that is reached by backprop
        :return: MSE_list: The list of MSE at the last stage
        :param FF(forward_filtering): [default to be true for historical reason] The flag to control whether use forward filtering or not
        """

        # expand the target spectra to eval batch size
        target_spectra_expand = target_spectra.expand([self.flags.population, -1])

        self.algorithm.set_pop_and_target(target_spectra_expand)
        for i in range(self.flags.generations):
            logit = self.algorithm.evolve()
            loss = self.make_loss(logit,target_spectra_expand)

        good_index = torch.argmin(loss,dim=0).cpu().data.numpy()
        geometry_eval_input = self.algorithm.generation.cpu().data.numpy()

        if save_all:  # If saving all the results together instead of the first one
            saved_model_str = self.saved_model.replace('/', '_')
            Ypred_file = os.path.join(save_dir, 'test_Ypred_point{}.csv'.format(saved_model_str))
            Xpred_file = os.path.join(save_dir, 'test_Xpred_point{}.csv'.format(saved_model_str))
            if self.flags.data_set != 'Yang':  # This is for meta-meterial dataset, since it does not have a simple simulator
                # 2 options: simulator/logit
                Ypred = simulator(self.flags.data_set, geometry_eval_input)
                if not save_Simulator_Ypred:  # The default is the simulator Ypred output
                    Ypred = logit.cpu().data.numpy()
                if len(np.shape(Ypred)) == 1:  # If this is the ballistics dataset where it only has 1d y'
                    Ypred = np.reshape(Ypred, [-1, 1])
                with open(Xpred_file, 'a') as fxp, open(Ypred_file, 'a') as fyp:
                    np.savetxt(fyp, Ypred[good_index, :])
                    np.savetxt(fxp, geometry_eval_input[good_index, :])

            else:
                with open(Xpred_file, 'a') as fxp:
                    np.savetxt(fxp, geometry_eval_input[good_index, :])


        ###################################
        # From candidates choose the best #
        ###################################
        Ypred = logit.cpu().data.numpy()

        if len(np.shape(Ypred)) == 1:  # If this is the ballistics dataset where it only has 1d y'
            Ypred = np.reshape(Ypred, [-1, 1])

        # calculate the MSE list and get the best one
        MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
        best_estimate_index = np.argmin(MSE_list)
        # print("The best performing one is:", best_estimate_index)
        Xpred_best = np.reshape(np.copy(geometry_eval_input[best_estimate_index, :]), [1, -1])
        if save_Simulator_Ypred and self.flags.data_set != 'Yang':
            Ypred = simulator(self.flags.data_set, geometry_eval_input)
            if len(np.shape(Ypred)) == 1:  # If this is the ballistics dataset where it only has 1d y'
                Ypred = np.reshape(Ypred, [-1, 1])
        Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])

        return Xpred_best, Ypred_best, MSE_list



class GA(object):
    def __init__(self,flags,model,loss_fn):
        ' Initialize general GA parameters '
        self.n_params = flags.linear[0]
        self.data_set = flags.data_set
        self.n_elite = flags.elitism
        self.k = flags.k
        self.n_pop = flags.population
        self.n_kids = flags.population - flags.elitism
        self.n_pairs = int(math.ceil(self.n_kids/2))
        self.mut = flags.mutation
        self.cross_p = flags.crossover
        self.selector = self.make_selector(flags.selection_operator)
        self.X_op = self.make_X_op(flags.cross_operator)
        self.device = torch.device('cpu' if flags.use_cpu_only else 'cuda')
        self.loss_fn = loss_fn
        self.eval = False
        self.span,self.lower,self.upper = self.get_boundary_lower_bound_upper_bound()

        self.model = model
        self.calculate_spectra = self.sim if self.eval else self.model
        self.target = None
        self.generation = None
        self.fitness = None
        self.p_child = None

    def get_boundary_lower_bound_upper_bound(self):
        if self.data_set == 'Yang':
            return torch.tensor([2.272, 2.272, 2.272, 2.272, 2, 2, 2, 2], device=self.device,requires_grad=False), \
                   torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1],device=self.device,requires_grad=False), \
                   torch.tensor([1.272, 1.272, 1.272, 1.272, 1, 1, 1, 1],device=self.device,requires_grad=False)
        else:
            return torch.tensor(self.n_params*[2],device=self.device,requires_grad=False),\
                   torch.tensor(self.n_params*[-1],device=self.device,requires_grad=False), \
                   torch.tensor(self.n_params*[1],device=self.device,requires_grad=False)

    def initialize_in_range(self, n_samples):
        return torch.rand(n_samples,self.n_params,device=self.device,requires_grad=False)*self.span + self.lower

    def sim(self,X):
        return torch.from_numpy(simulator(self.data_set,X.cpu().data.numpy())).float().cuda()

    def set_pop_and_target(self, trgt):
        ' Initialize parameters that change with new targets: fitness, target, generation '
        self.target = trgt
        self.target.requires_grad = False
        self.generation = self.initialize_in_range(self.n_pop)
        self.p_child = torch.empty(2*self.n_pairs,self.n_params, device=self.device, requires_grad=False)
        self.fitness = torch.empty(self.n_pop, device=self.device, requires_grad=False)

    def make_selector(self, S_type):
        ' Returns selection operator used to select parent mating pairs '
        if S_type == 'roulette':
            return self.roulette
        elif S_type == 'tournament':
            return self.tournament
        elif S_type == 'decimation':
            return self.decimation
        else:
            raise(Exception('Selection Operator improperly configured'))

    def make_X_op(self, X_type):
        ' Returns crossover operator '
        if X_type == 'uniform':
            return self.u_cross
        elif X_type == 'single-point':
            return self.s_cross
        else:
            raise (Exception('Crossover Operator improperly configured'))

    def roulette(self):
        ' Performs roulette-wheel selection from self.generation using self.fitness '
        mock_fit = 1/self.fitness
        total = torch.sum(mock_fit)
        mock_fit = mock_fit/total

        r_wheel = torch.cumsum(mock_fit,0)
        spin_values = torch.rand(2*self.n_pairs,1, device=self.device, requires_grad=False)

        r_wheel = r_wheel.unsqueeze(0).expand(2*self.n_pairs,-1)
        dist = r_wheel - spin_values
        dist[torch.heaviside(-dist,torch.tensor(0.0, device=self.device,requires_grad=False)).bool()] = 1
        idxs = torch.argmin(dist, dim=1)

        self.p_child = self.generation[idxs,:].clone()
        self.p_child = self.p_child.unsqueeze(1).view(self.n_pairs,2,self.n_params)

    def decimation(self):
        ' Performs population decimation selection from self.generation assuming it is sorted by fitness '
        self.p_child = self.generation[torch.randint(self.k,(2*self.n_pairs,),device=self.device,requires_grad=False),:].clone()
        self.p_child = self.p_child.unsqueeze(1).view(self.n_pairs,2,self.n_params)

    def tournament(self):
        ' Performs tournament-style selection from self.generation assuming it is sorted by fitness '
        for row in range(2*self.n_pairs):
            self.p_child[row] = self.generation[torch.min(torch.randperm(self.n_pop,device=self.device,requires_grad=False)[:self.k])]
        self.p_child = self.p_child.clone().unsqueeze(1).view(self.n_pairs,2,self.n_params)

    def u_cross(self):
        ' Performs uniform crossover given pairs tensor arranged sequentially in parent-pairs '
        x_mask = torch.heaviside(torch.rand(self.n_pairs, device=self.device, requires_grad=False) - self.cross_p,
                                 torch.tensor(1.0, device=self.device, requires_grad=False)).bool()
        selectX = self.p_child[x_mask]
        n_selected = selectX.shape[0]
        site_mask = torch.heaviside(torch.rand(n_selected,self.n_params, device=self.device, requires_grad=False)-0.5,
                                    torch.tensor(1.0, device=self.device, requires_grad=False)).bool()

        parentC = self.p_child.clone()

        for i in range(n_selected):
            p0 = self.p_child[i][0]
            p0c = parentC[i][0]
            p1 = self.p_child[i][1]
            p1c = parentC[i][1]

            p0[site_mask[i]] = p1c[site_mask[i]]
            p1[site_mask[i]] = p0c[site_mask[i]]

    def s_cross(self):
        ' Performs single-point crossover given pairs tensor arranged sequentially in parent-pairs '
        x_mask = torch.heaviside(torch.rand(self.n_pairs, device=self.device, requires_grad=False) - self.cross_p,
                                 torch.tensor(1.0, device=self.device, requires_grad=False)).bool()
        selectX = self.p_child[x_mask]
        siteX = torch.randint(1, self.n_params, (selectX.shape[0],), device=self.device, requires_grad=False)
        n_selected = selectX.shape[0]

        parentC = self.p_child.clone()

        for i in range(n_selected):
            p0 = self.p_child[i][0]
            p0c = parentC[i][0]
            p1 = self.p_child[i][1]
            p1c = parentC[i][1]

            p0[siteX[i]:] = p1c[siteX[i]:]
            p1[siteX[i]:] = p0c[siteX[i]:]

    def mutate(self):
        ' Performs single-point random mutation given children tensor '
        self.p_child = self.p_child.view(2*self.n_pairs,self.n_params)
        param_prob = torch.heaviside(self.mut - self.initialize_in_range(2*self.n_pairs),
                                     torch.tensor(1.0,device=self.device,requires_grad=False)).bool()
        self.p_child[param_prob] = (torch.rand_like(self.p_child[param_prob]) - 0.5)/0.5

    def evolve(self):
        ' Function does the genetic algorithm. It evaluates the next generation given previous one'

        if self.target is None:
            raise(Exception('Set target spectra before running the GA'))

        # Evaluate fitness of current population
        logit = self.calculate_spectra(self.generation)
        self.fitness = self.loss_fn(logit,self.target)

        # Select parents for mating and sort individuals in generation
        self.fitness, sorter = torch.sort(self.fitness,descending=False)
        self.generation = self.generation[sorter, :]
        self.selector()

        # Do crossover followed by mutation to create children from parents
        self.X_op()
        self.mutate()

        # Combine children and elites into new generation
        self.generation[self.n_elite:] = self.p_child[:self.n_kids]
        return logit