# Get the out of distribution from the original generated extended geometries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 


def read_extend(dataset):
    if dataset == 'Yang':
        data_x = pd.read_csv('Yang_sim/dataIn/data_x.csv', header=None, sep=' ').values
        data_y = pd.read_csv('Yang_sim/dataIn/data_y.csv', header=None, sep=' ').values
    elif dataset == 'Chen':
        data_x = pd.read_csv('Chen/data_x_extended.csv', header=None, sep=',').values
        data_y = pd.read_csv('Chen/data_y_extended.csv', header=None, sep=',').values
    elif dataset == 'Peurifoy':
        data_x = pd.read_csv('Peurifoy/data_x_extended.csv', header=None, sep=',').values
        data_y = pd.read_csv('Peurifoy/data_y_extended.csv', header=None, sep=',').values
    else:
        print("Your dataset is incorrect! Aborting")
        quit()
    return data_x, data_y


def get_out_of_range(dataset):
    print('dataset is', dataset)
    # get x and y data
    data_x, data_y = read_extend(dataset)
    # Get the upper bound of the extended dataset so that to generate the smaller test set
    if dataset == 'Yang':
        x_bound = 0.9
    elif dataset == 'Chen':
        x_bound = 50
    elif dataset == 'Peurifoy':
        x_bound = 70
    else:
        print("Your dataset is incorrect! Aborting")
        quit()
    
    print('max of data_x', np.max(data_x))
    # Check if all the dimensions are larger than that
    data_x -= x_bound
    out_of_bound = np.any(data_x > 0, axis=1)
    print('shape of out_of_bound = ', np.shape(out_of_bound))
    print('ratio of out_of_bound = ', np.sum(out_of_bound)/len(out_of_bound))

    # After getting the out_of_range flags, use them to output the out_of_range x and ys
    out_of_range_x = data_x[out_of_bound, :]
    out_of_range_y = data_y[out_of_bound, :]

    # Save the out of range data for further evaluation
    
    

if __name__ == '__main__':
    dataset_list = ['Yang','Chen','Peurifoy']
    #dataset = 'Peurifoy'
    #dataset = 'Yang'
    #dataset = 'Chen'
    #data_x, data_y = read_extend(dataset)
    for dataset in dataset_list:
        get_out_of_range(dataset)
