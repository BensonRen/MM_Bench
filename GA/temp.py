import flag_reader
import torch
import sys
import numpy as np
sys.path.append('C:\\Users\\Lab User\\Desktop\\Ashwin\\MM_Bench\\utils')
from utils.helper_functions import save_flags, load_flags
from utils.evaluation_helper import plotMSELossDistrib as pl


print('\n'.join(sys.path))
flags = load_flags('.\\models\\Yang_sim_best_model')
dir = 'data/sweep02/Yang_sim_1000/'
pl(dir+'test_Ypred_Yang_sim_best_model.csv',dir+'test_Ytruth_Yang_sim_best_model.csv',flags,save_dir=dir)


'''
flags = flag_reader.read_flag()
save_flags(flags,'.\\models\\peurifoy')
'''
'''
flags = load_flags('.\\models\\Yang_sim_best_model')
flags.data_dir = '../Data/'
save_flags(flags,'.\\models\\Yang_sim_best_model')
flags = load_flags('.\\models\\Yang_sim_best_model')
with open('.\\models\\Yang_sim_best_model\\parameters.txt','w') as file:
    print(vars(flags), file=file)
    print(vars(flags))
'''
'''
flags = load_flags('.\\models\\peurifoy')
flags.linear = [8,400,400,400,201]
flags.ga_eval = False
save_flags(flags,'.\\models\\peurifoy')
'''

'''
from ga import GA

flags = flag_reader.read_flag()
flags.population = 100
flags.crossover = 0.7
flags.mutation = 0.01
flags.elitism = 1
flags.generations = int(6000/flags.population)
flags.selection_operator = 'roulette'

def loss(X,l):
    return X

def fake_model(X):
    res = 1
    for i in range(X.shape[1]):
        res = res * torch.abs(torch.sin(X[:, i] - 2) / (X[:, i] - 2))
    return res

ga = GA(flags, fake_model, loss)
ga.set_pop_and_target(torch.ones(8,device='cuda'))

f = open('roulette.csv','w')
g = open('roulette-10.csv','w')
h = open('tournament.csv','w')
j = open('tournament-10.csv','w')
f.close()
g.close()
h.close()
j.close()

with open('roulette.csv','a') as f:
    b_val = 0
    for g in range(flags.generations):
        l = ga.evolve()
        val = l.cpu().data.numpy()[0]
        if val > b_val:
            b_val = val
        f.write("{},{}\n".format(g*flags.population, b_val))
        print(g)
    print(ga.generation[0])

flags.selection_operator = 'tournament'
ga = GA(flags, fake_model, loss)
ga.set_pop_and_target(torch.ones(8,device='cuda'))
with open('tournament.csv','a') as f:
    b_val = 0
    for g in range(flags.generations):
        l = ga.evolve()
        val = l.cpu().data.numpy()[0]
        if val > b_val:
            b_val = val
        f.write("{},{}\n".format(g*flags.population, b_val))
        print(g)
    print(ga.generation[0])

flags.selection_operator = 'roulette'
flags.elitism = 90
flags.generations = int(6000/(flags.population - flags.elitism))
ga = GA(flags, fake_model, loss)
ga.set_pop_and_target(torch.ones(8,device='cuda'))
with open('roulette-10.csv','a') as f:
    b_val = 0
    for g in range(flags.generations):
        l = ga.evolve()
        val = l.cpu().data.numpy()[0]
        if val > b_val:
            b_val = val
        if g%10 == 0:
            f.write("{},{}\n".format(g*(flags.population - flags.elitism), b_val))
        print(g)
    print(ga.generation[0])

flags.selection_operator = 'tournament'
flags.elitism = 90
ga = GA(flags, fake_model, loss)
ga.set_pop_and_target(torch.ones(8,device='cuda'))
with open('tournament-10.csv','a') as f:
    b_val = 0
    for g in range(flags.generations):
        l = ga.evolve()
        val = l.cpu().data.numpy()[0]
        if val > b_val:
            b_val = val
        if g % 10 == 0:
            f.write("{},{}\n".format(g*(flags.population - flags.elitism), b_val))
        print(g)
    print(ga.generation[0])
'''
