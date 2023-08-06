import numpy as np
import pandas as pd

import pickle
from scipy.spatial import distance
from scipy.optimize import curve_fit

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample

import keras
from keras.models import Sequential
from keras.layers import Dense

import itertools
from itertools import zip_longest
def avg(x):
    '''
    Calculating average value of a list, in which 'None' records were ignored
    :param x: input list
    :returns: average value calculated
    '''
    x = [i for i in x if i is not None]
    return sum(x, 0.0) / len(x)


def generate_dataset(dfs, K):
    '''
    Generate desired dataset for dataframes collected in a scenario.
    In the original dataframe are the records of coordinates '(x, y, z)' for each pedestrians on different time frames, 
    from which records of distances to k-nearest-neighburs (X) and velocity (y) could be calculated.

    :param dfs: a list of dataframes collected within a certain scenario
    :param K: number of nearest neighbors to keep in the dataset
    :returns: a tuple of X (input) and y (output) for the dataset
    '''
    dataset_X = []
    dataset_y = []

    # for each given dataframe in the scenario
    for data in dfs:
        # generate dictionaries of id, coordinates and distance matrix on each time frame 
        id_frame_dict = {}
        coords_frame_dict = {}
        dist_frame_dict = {}
        for frame in range(min(data['frame']), max(data['frame'])+1):
            tmp = data[data['frame']==frame]
            ids = list(tmp['id'])
            coords = list(tmp['coords'])
            
            id_frame_dict[frame] = ids
            coords_frame_dict[frame] = dict(zip(ids, coords))
            dist_frame_dict[frame] = dict(zip(ids, distance.cdist(coords, coords, 'euclidean')))
        
        # generate dictionaries of the list of distance to neighbors and velocity across time frames on each id
        dist_list_id_dict = {}
        velocity_list_id_dict = {}
        for id in range(min(data['id']), max(data['id'])+1):
            tmp = data[data['id']==id]
            dist_list = []
            velocity_list = []
            for frame in range(min(tmp['frame']), max(tmp['frame'])):
                dist_list.append(sorted(dist_frame_dict[frame][id]))
                velocity_list.append(np.linalg.norm(np.subtract(coords_frame_dict[frame][id], coords_frame_dict[frame+1][id])))
            dist_list_id_dict[id] = dist_list
            velocity_list_id_dict[id] = velocity_list

        # generate record lists of X (distance to neighbors) and y (velocity) as the model's input and output
        data_X = []
        data_y = []
        for id in range(min(data['id']), max(data['id'])+1):
            data_X += dist_list_id_dict[id]
            data_y += velocity_list_id_dict[id]

        # select and prune the records into distances to K nearest neighbors
        for X, y in list(zip(data_X, data_y)):
            if len(X) > K:
                dataset_X.append([i / 10 for i in X[1:K]])
                dataset_y.append(y / 10)

    return (dataset_X, dataset_y)
        
def weidmann_model(x, v0, l, T):
    '''
    Weidmann Model Function
    :param x: mean distance to k nearest neighbors
    :param v0: desired speed
    :param l: pedestrian size
    :param T: time gap
    '''
    return v0 * (1 - np.exp((l - x) / v0 / T))


def measure(y_test, y_pred):
    '''
    Measure the performance of the model by calculating MAE, MSE and R2 score.
    :param y_test, y_pred: desired output and prediction to compare
    '''
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {'MAE': mae, 'MSE': mse, 'R2': r2}


def NN(layers, X_train, y_train, n_splits=5, epochs=20):
    '''
    Implement and train a Neural Network with the assigned structure.
    :param layers: number of nodes on each hidden layer, representing the structure of the Neural Network
    :param X_train, y_train: training set input and output
    :param n_splits: K-fold validation parameter
    :param epochs: number of epochs in total during the training process
    '''
    model = Sequential()
    
    model.add(Dense(layers[0], input_shape=(X_train.shape[1],), activation='relu'))
    if len(layers)>=2:
        for i in layers[1:]:
            model.add(Dense(i, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []
    count = 0
    for train_index, val_index in kfold.split(X_train):
        count += 1
        print("Training Process:", count, '/', n_splits, "Fold")
        X_train_bootstrap, X_val_bootstrap = X_train[train_index], X_train[val_index]
        y_train_bootstrap, y_val_bootstrap = y_train[train_index], y_train[val_index]
        
        X_train_boostrap_resampled, y_train_bootstrap_resampled = resample(X_train_bootstrap, y_train_bootstrap)
        model.fit(X_train_boostrap_resampled, y_train_bootstrap_resampled, epochs=int(epochs/n_splits), batch_size=32, verbose=2)
        score = model.evaluate(X_val_bootstrap, y_val_bootstrap, verbose=0)
        scores.append(score)
        
    print("\nK-Fold Validation MSE: %.2f" % np.mean(scores))
    return model