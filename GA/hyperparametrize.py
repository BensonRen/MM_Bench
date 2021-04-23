import numpy as np
import os
import GA_manager from ga as GA_obj
import evaluate_from_model from evaluate
from utils.helper_functions import load_flags


# Goal is to have functions that if arranged correctly in another folder will allow the management and running of GAs

# GA Run PseudoCode
#1. Initialize a population of objects
#2. Evaluate the objects (store evaluation in files)
#3. Pick elite objects to be parents
#4. Create and evaluate a new set of child objects
#5. Filter new generation to be best of both parent and child object sets
#6. Repeat

# def add_to_plot()
# def initialize()
# Distirbute over GPUs?

class GA_HyperSweep(object):
    def __init__(self):
        self.lower_bound = np.array([0,0,0.6,0.01,1])
        self.upper_bound = np.array([3,2,0.8,0.1,1025])
        self.Selection_Ops = ['tournament','roulette','decimation']
        self.Cross_Ops = ['single-point','uniform']

        self.dataset = 'Peurifoy' #'Chen', 'Yang_sim'
        self.population = 15
        self.elites = 3
        self.Xp = 0.8
        self.mut = 0.1

        self.path = self.build_path()

        with open(os.path.join(self.path,'sweep-parameters.txt'),'w') as f:
            print(vars(self),file=f)

    def build_path(self):
        root = os.path.join('data','GA_sweep',self.dataset)
        root_directories = os.listdir(root)

        cnt = 0
        for directory in root_directories:
            if directory.find('sweep') != -1:
                cnt += 1

        self.path = os.path.join(root,'sweep{}'.format(cnt))
        os.mkdir(self.path)

    def generate_new_GA_obj(self):
        #Selection_Op x3, Cross_Op x2, Crossover 0.6-0.8, Mutation 0.01-0.1, Elitism=k: 1-1024
        gene = []
        protogene = np.random.rand(5)*(self.upper_bound-self.lower_bound) + self.lower_bound

        gene[0] = self.Selection_Ops[np.floor(protogene[0])]
        gene[1] = self.Cross_Ops[np.floor(protogene[1])]
        gene[2] = protogene[2]
        gene[3] = protogene[3]
        gene[4] = np.floor(protogene[4])

        dir = self.dataset + '_best_model'
        flags = load_flags(os.path.join("models", dir))
        flags.data_set = self.dataset
        flags.population = 2048
        flags.generations = 100
        flags.xtra = 300

        flags.selection_operator = gene[0]
        flags.cross_operator = gene[1]
        flags.crossover = gene[2]
        flags.mutation = gene[3]
        flags.elitism = gene[4]
        flags.k = gene[4]

        flags.eval_model = dir
        flags.save_to = '_'.join(map(str,[self.path]+gene))
        flags.ga_eval = False

        return flags

    def initialize_population(self):
        population = [self.generate_new_GA_obj() for i in range(self.population)]
        return population

    def run_and_sort_population(self, population):
        old_model_directories = os.listdir(self.path)

        fitness = np.empty(len(population))

        for i,m in enumerate(population):
            if not m.save_to in old_model_directories:
                evaluate_from_model(m.eval_model, preset_flags=m, save_Simulator_Ypred=False)

            with open(os.path.join(m.save_to,'best_loss.txt'),'r') as f:
                fitness[i] = float(f.read())

        # Sort the population
        sorter = np.argsort(fitness)
        population = population[sorter]

        # Filter if your population is too big (i.e. if this is used after children are evaluated)
        population = population[:self.population]

        return population,fitness

    def create_children(self, population, fitness):
        children = []
        #Do a roulette selection of parents to make children
        return children

    def start(self):
        self.initialize_population()
        fitness = np.empty(self.population)

        while():
            pass
            # run and sort population
            # make kids and add them to population
