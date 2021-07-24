import os
import sys

sys.path.append('C:\\Users\\Lab User\\Desktop\\Ashwin\\MM_Bench\\utils')

import numpy as np
import torch
import matplotlib.pyplot as plt
import flag_reader
from Forward.class_wrapper import Network
from Forward.model_maker import Forward
from utils import data_reader
from utils.helper_functions import save_flags, load_flags, simulator

sets = ["Peurifoy","Chen","Yang_sim"]
folder = 'loglog-scatter-vline-under-20-mse'

if not os.path.exists('one-to-many/'+folder):
    os.mkdir('one-to-many/'+folder)

for dset in sets:
    flags = flag_reader.read_flag()
    flags.data_set = dset
    flags.model_name = flags.data_set.lower()
    flags.eval_model = flags.model_name

    train_loader, test_loader = data_reader.read_data(flags,eval_data_all=True)
    geo = None
    spect = None

    for g,s in test_loader:
        if geo is None:
            geo = g.data.numpy()
            spect = s.data.numpy()
        else:
            geo = np.vstack((geo,g.data.numpy()))
            spect = np.vstack((spect,s.data.numpy()))

    geo = torch.from_numpy(np.array(geo)).cuda()
    spect = torch.from_numpy(np.array(spect)).cuda()

    for i in range(len(geo)):
        T_spect = spect[i]
        T_geo = geo[i]

        spect_mse = torch.mean(torch.pow(T_spect - spect,2),dim=1)
        geo_mse = torch.mean(torch.pow(T_geo - geo,2),dim=1)

        if i == 0:
            spect_massive = spect_mse
            geometric_massive = geo_mse
        else:
            spect_massive = torch.vstack((spect_massive,spect_mse))
            geometric_massive = torch.vstack((geometric_massive,geo_mse))

        if i == 999:
            print(spect_massive.shape)
            print(geometric_massive.shape)
            break


    invs = load_flags('../Forward/models/'+dset+'_invs_best')
    xlim = invs.best_validation_loss
    fwd = load_flags('models/'+dset+'_best_model')
    ylim = fwd.best_validation_loss

    invs_model = Network(Forward,invs,train_loader,test_loader,saved_model='../Forward/models/'+dset+'_invs_best',inference_mode=True)
    invs_model.load()
    print(invs_model.ckpt_dir)
    invs_model = invs_model.model
    invs_model.cuda().eval()
    #geo_mse_model = torch.mean(torch.pow(invs_model(spect) - geo, 2),dim=1)

    # Spectral MSE Threshold
    x = np.expand_dims(np.linspace(0,spect_massive.max().item(),100),1)
    x = torch.from_numpy(x).cuda()

    # Geometric MSE Threshold
    y = np.linspace(0, geometric_massive.max().item(), 100)
    y = torch.from_numpy(y).cuda()

    z_t = torch.empty(100,100).cuda()

    for i,spec_diff_lim in enumerate(x):
        # Find spectrum that are similar in MSE UNDER spec_diff_lim
        SPEC_SIMILARITY = spect_massive < spec_diff_lim
        SPEC_SIMILARITY = SPEC_SIMILARITY.float()

        for j,geo_diff_lim in enumerate(y):
            # Find geometry that are dissimilar im MSE OVER
            GEO_DISSIMILARITY = geometric_massive > geo_diff_lim
            GEO_DISSIMILARITY = GEO_DISSIMILARITY.float()

            COUNT_SPEC_GEO_CONDITIONS = torch.sum(torch.dot(SPEC_SIMILARITY.flatten(),GEO_DISSIMILARITY.flatten()))
            #if COUNT_SPEC_GEO_CONDITIONS == 0:
            #    COUNT_SPEC_GEO_CONDITIONS = float('nan')
            #print(SPEC_SIMILARITY.sum(),GEO_DISSIMILARITY.sum(),COUNT_SPEC_GEO_CONDITIONS)

            z_t[j,i] = COUNT_SPEC_GEO_CONDITIONS

    x_t, y_t = np.meshgrid(x.cpu().data.numpy(), y.cpu().data.numpy())
    z_t = z_t.cpu().data.numpy()
    #z_t[z_t == 0] = np.nan
    f = plt.figure()

    print('X:\n',x_t,'\n\nY:',y_t,'\n\nZ:',z_t)

    ax = plt.axes(projection='3d')
    #ax.scatter3D(x_t/spect_massive.max().item(),y_t/geometric_massive.max().item(),z_t)
    ax.plot_wireframe(x_t/spect_massive.max().item(),y_t/geometric_massive.max().item(),z_t)
    ax.view_init(-140,90)
    ax.set_ylabel("Geometric threshold of dissimilarity")
    ax.set_xlabel("Spectral threshold of similarity")
    #ax.set_zlabel("Number of spectra meeting both conditions (clipped at 20)")
    plt.savefig(dset+'_test.png',transparent=False)



'''for dset in sets:
    # flags = flag_reader.read_flag()
    # flags.data_set = dset
    # flags.model_name = flags.data_set.lower()
    # flags.eval_model = flags.model_name
    #
    # train_loader, test_loader = data_reader.read_data(flags,eval_data_all=True)
    # geo = None
    # spect = None
    #
    # for g,s in test_loader:
    #     if geo is None:
    #         geo = g.data.numpy()
    #         spect = s.data.numpy()
    #     else:
    #         geo = np.vstack((geo,g.data.numpy()))
    #         spect = np.vstack((spect,s.data.numpy()))
    #
    # geo = torch.from_numpy(np.array(geo)).cuda()
    # spect = torch.from_numpy(np.array(spect)).cuda()
    #
    # for i in range(len(geo)):
    #     T_spect = spect[i]
    #     T_geo = geo[i]
    #
    #
    #     spect_mse = torch.mean(torch.pow(T_spect - spect,2),dim=1)
    #     geo_mse = torch.mean(torch.pow(T_geo - geo,2),dim=1)
    #
    #     if i == 0:
    #         mse_massive = spect_mse
    #         geometric_massive = geo_mse
    #     else:
    #         mse_massive = torch.vstack((mse_massive,spect_mse))
    #         geometric_massive = torch.vstack((geometric_massive,geo_mse))
    #
    #     if i == 999:
    #         print(mse_massive.shape)
    #         print(geometric_massive.shape)
    #         break
    #
    #
    # invs = load_flags('../Forward/models/'+dset+'_invs_best')
    # xlim = invs.best_validation_loss
    # fwd = load_flags('models/'+dset+'_best_model')
    # ylim = fwd.best_validation_loss
    #
    # invs_model = Network(Forward,invs,train_loader,test_loader,saved_model='../Forward/models/'+dset+'_invs_best',inference_mode=True)
    # invs_model.load()
    # print(invs_model.ckpt_dir)
    # invs_model = invs_model.model
    # invs_model.cuda().eval()
    # geo_mse_model = torch.mean(torch.pow(invs_model(spect) - geo, 2),dim=1)
    #
    # x = np.linspace(0,mse_massive.max().item(),100)
    # y = np.empty(len(x))
    # for i,spec_diff_lim in enumerate(x):
    #     GEO_LOWER = geometric_massive < geo_mse_model
    #     SPEC_HIGHER = mse_massive < spec_diff_lim
    #
    #     GEO_LOWER = GEO_LOWER.float()
    #     SPEC_HIGHER = SPEC_HIGHER.float()
    #
    #     GEO_LOWER_AND_SPEC_HIGHER = torch.dot(GEO_LOWER.flatten(), SPEC_HIGHER.flatten())
    #     y[i] = GEO_LOWER_AND_SPEC_HIGHER.sum().item()
    y = np.genfromtxt(dset+'_daty.csv')
    x = np.genfromtxt(dset+'_datx.csv')

    #f = plt.figure()
    plt.scatter(x,y,label=dset)
    #plt.scatter(y,x/mse_massive.max().item())
    #np.savetxt(dset+'_daty.csv',x/mse_massive.max().item())
    #np.savetxt(dset + '_datx.csv',y)
    #plt.gca().invert_xaxis()

plt.legend()
plt.xscale('log')
plt.savefig('test.png',transparent=False)'''

'''
    geo_mse_model = geo_mse_model.cpu().data.numpy()
    print("BVL Geo:\t",invs.best_validation_loss)
    for i in range(5):
        x = geometric_massive[i].cpu().data.numpy()
        y = mse_massive[i].cpu().data.numpy()
        print(geo_mse_model.shape, geo_mse_model[i])
        xlim = geo_mse_model[i]

        sorter = np.argsort(y)

        x = x[sorter[0:20]]
        y = y[sorter[0:20]]

        f = plt.figure()
        ax = f.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.scatter(x,y)
        plt.axvline(xlim,0,1)
        plt.text(xlim,2*ylim,"MSE={:.3e}".format(xlim),rotation=90,va='center')

        plt.title("Geometric and Spectral MSE of one TS vs. all other datapoints")
        plt.xlabel('MSE Geometry')
        plt.ylabel('MSE Spectra')
        plt.axhline(ylim,0,1)
        plt.text(2*xlim,ylim,"MSE={:.3e}".format(xlim),ha='center')
        plt.savefig('one-to-many/{}/{}_{}{}.png'.format(folder,dset,folder,i))

    # f = plt.figure()
    # plt.hist2d(x,y)
    # plt.savefig('one-to-many/{}_hist2d{}.png'.format(dset,i))
    
    '''










