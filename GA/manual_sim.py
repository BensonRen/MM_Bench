import os
import sys
sys.path.append('C:\\Users\\Lab User\\Desktop\\Ashwin\\MM_Bench\\utils')
import numpy as np
import flag_reader
import torch
from ga import GA_manager as GA
from Forward.class_wrapper import Network

from model_maker import Net
from utils import data_reader
from utils.helper_functions import load_flags
from utils.helper_functions import save_flags, load_flags, simulator
from utils.evaluation_helper import plotMSELossDistrib as pl

def sim_one(dirx,dset,plot=False):
    flags = flag_reader.read_flag()
    flags.data_set = dset
    flags.model_name = flags.data_set.lower()
    flags.eval_model = flags.model_name

    if '.csv' in dirx:
        fxp = dirx
        fyp = fxp.replace('Xpred','Ypred')
        fyt = fxp.replace('Xpred','Ytruth')
    else:
        fxp = dirx+'test_Xpred_'+flags.data_set+'_best_model.csv'
        fyp = dirx+'test_Ypred_'+flags.data_set+'_best_model.csv'
        fyt = dirx+'test_Ytruth_'+flags.data_set+'_best_model.csv'

    xmat = np.genfromtxt(fxp,delimiter=' ')
    ypred = simulator(flags.data_set, xmat)
    np.savetxt(fyp, ypred, delimiter=' ')

    if plot:
        pl(fyp, fyt, flags, save_dir=dirx)

def sim_all_in_folder(dirx,dset,excp_str=None,add_str=None,plot=False):
    files = os.listdir(dirx)
    for f in files:

        if 'Yang_sim' in f:
            os.remove(os.path.join(dirx,f))
            continue

        if 'Xpred' in f:
            if f.replace('Xpred','Ypred') in files:
                continue
            if add_str is None or not (add_str in f):
                continue
            if not (excp_str is None) and excp_str in f:
                continue
            print(os.path.join(dirx,f))
            sim_one(os.path.join(dirx,f),dset,plot=plot)

def evaluate_forward_model(dirx,n_samples,invs=False):
    print("DIRECTORY: ",dirx)
    flags = load_flags(dirx)
    flags.batch_size = 1
    train_loader, test_loader = data_reader.read_data(flags)
    GEN = GA(flags, train_loader, test_loader, inference_mode=True, saved_model=dirx)

    GEN.model.eval()
    avg_mse, avg_mre, avg_rse = 0,0,0
    for i, (g,s) in enumerate(test_loader):
        if invs:
            z = s
            s = g
            g = z

        g = g.cuda()
        s = s.cuda()
        ps = GEN.model(g)

        if invs:
            pg = ps
            z = g
            g = s
            s = z
            ps = simulator(flags.data_set, pg.cpu().data.numpy())
            ps = torch.from_numpy(ps).cuda()

        mse = torch.nn.functional.mse_loss(s, ps)
        rse = torch.sqrt(torch.sum(torch.pow(s - ps, 2))) / torch.sqrt(torch.sum(torch.pow(s, 2)))
        mre = torch.mean(torch.abs(torch.div(s - ps, s)))

        avg_mse += mse.item()
        avg_rse += rse.item()
        avg_mre += mre.item()

        if i==(n_samples - 1):
            print('BROKE at sample {}'.format(i))
            break

    avg_mse /= n_samples
    avg_mre /= n_samples
    avg_rse /= n_samples

    print("\nMSE:\t{}\nMRE:\t{}\nRSE:\t{}".format(avg_mse,avg_mre,avg_rse))

if __name__ == '__main__':
    evaluate_forward_model(os.path.join('../Forward/models','Yang_sim_invs_best'),1000000,invs=True)

    # for d1 in ('res_rev','unres_rev'):
    #     for dset in ('Chen','Peurifoy'):
    #         sim_all_in_folder(os.path.join('multi_eval',d1,dset),dset,add_str='inference')

    #dset = 'Peurifoy'
    #dirx = 'multi_eval/unres/'+dset+'/'
    #sim_one(dirx,dset,plot=True)

