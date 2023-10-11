import numpy as np

from matplotlib import pyplot as plt

from Sensors.FlowerEchoSimulator.Spatializer import wrapToPi

import torch
import torch.nn as nn

class WrapToPiMSELoss(nn.Module):
    def __init__(self):
        super(WrapToPiMSELoss, self).__init__()
        
    def forward(self, prediction, target):
        return torch.mean((wrapToPi(prediction - target))**2)
    
class WrapToPiMAELoss(nn.Module):
    def __init__(self):
        super(WrapToPiMAELoss, self).__init__()
        
    def forward(self, prediction, target):
        return torch.mean(torch.abs(wrapToPi(prediction - target)))
    
class WrapToPiMAPELoss(nn.Module):
    def __init__(self):
        super(WrapToPiMAPELoss, self).__init__()
        
    def forward(self, prediction, target):
        return torch.mean(torch.abs(wrapToPi((prediction - target)/target)))
    
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        
    def forward(self, prediction, target):
        return torch.mean(torch.abs(prediction - target))
    
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()
        
    def forward(self, prediction, target):
        return torch.mean(torch.abs((prediction - target)/target))


def get_test_distance_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                  azimuth_range, orientation_range, n_samples=100):
    # filter X_test, y_distance_test, y_azimuth_test, y_orientation_test that has azimuth and orientation in the ranges
    find = np.where((y_azimuth_test >= azimuth_range[0]) & (y_azimuth_test < azimuth_range[1]) &
                    (y_orientation_test >= orientation_range[0]) & (y_orientation_test < orientation_range[1]))[0]
    return X_test[find][:n_samples], y_distance_test[find][:n_samples]
    # randomly select n_samples from the filtered data
    #return X_test[find][np.random.choice(find.shape[0], n_samples, replace=False)], y_distance_test[find][np.random.choice(find.shape[0], n_samples, replace=False)]

def get_test_azimuth_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                    distance_range, orientation_range, n_samples=100):
        # filter X_test, y_distance_test, y_azimuth_test, y_orientation_test that has azimuth and orientation in the ranges
        find = np.where((y_distance_test >= distance_range[0]) & (y_distance_test < distance_range[1]) &
                        (y_orientation_test >= orientation_range[0]) & (y_orientation_test < orientation_range[1]))[0]
        return X_test[find][:n_samples], y_azimuth_test[find][:n_samples]

def get_test_orientation_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                    distance_range, azimuth_range, n_samples=100):
        # filter X_test, y_distance_test, y_azimuth_test, y_orientation_test that has azimuth and orientation in the ranges
        find = np.where((y_distance_test >= distance_range[0]) & (y_distance_test < distance_range[1]) &
                        (y_azimuth_test >= azimuth_range[0]) & (y_azimuth_test < azimuth_range[1]))[0]
        return X_test[find][:n_samples], y_orientation_test[find][:n_samples]

def get_test_distance_from_ranges2(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                  distance_range, orientation_range, n_samples=100):
    # filter X_test, y_distance_test, y_azimuth_test, y_orientation_test that has azimuth and orientation in the ranges
    find = np.where((y_distance_test >= distance_range[0]) & (y_distance_test < distance_range[1]) &
                    (y_orientation_test >= orientation_range[0]) & (y_orientation_test < orientation_range[1]))[0]
    return X_test[find][:n_samples], y_distance_test[find][:n_samples]

def get_test_distance_from_ranges3(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                  distance_range, azimuth_range, n_samples=100):
    # filter X_test, y_distance_test, y_azimuth_test, y_orientation_test that has azimuth and orientation in the ranges
    find = np.where((y_distance_test >= distance_range[0]) & (y_distance_test < distance_range[1]) &
                    (y_azimuth_test >= azimuth_range[0]) & (y_orientation_test < azimuth_range[1]))[0]
    return X_test[find][:n_samples], y_distance_test[find][:n_samples]

def get_test_azimuth_from_ranges2(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                    distance_range, azimuth_range, n_samples=100):
        # filter X_test, y_distance_test, y_azimuth_test, y_orientation_test that has azimuth and orientation in the ranges
        find = np.where((y_distance_test >= distance_range[0]) & (y_distance_test < distance_range[1]) &
                        (y_azimuth_test >= azimuth_range[0]) & (y_azimuth_test < azimuth_range[1]))[0]
        return X_test[find][:n_samples], y_azimuth_test[find][:n_samples]

def get_test_azimuth_from_ranges3(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                    orientation_range, azimuth_range, n_samples=100):
        # filter X_test, y_distance_test, y_azimuth_test, y_orientation_test that has azimuth and orientation in the ranges
        find = np.where((y_azimuth_test >= azimuth_range[0]) & (y_azimuth_test < azimuth_range[1]) &
                        (y_orientation_test >= orientation_range[0]) & (y_orientation_test < orientation_range[1]))[0]
        return X_test[find][:n_samples], y_azimuth_test[find][:n_samples]

def get_test_orientation_from_ranges2(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                    distance_range, orientation_range, n_samples=100):
        # filter X_test, y_distance_test, y_azimuth_test, y_orientation_test that has azimuth and orientation in the ranges
        find = np.where((y_distance_test >= distance_range[0]) & (y_distance_test < distance_range[1]) &
                        (y_orientation_test >= orientation_range[0]) & (y_orientation_test < orientation_range[1]))[0]
        return X_test[find][:n_samples], y_orientation_test[find][:n_samples]

def get_test_orientation_from_ranges3(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                    azimuth_range, orientation_range, n_samples=100):
        # filter X_test, y_distance_test, y_azimuth_test, y_orientation_test that has azimuth and orientation in the ranges
        find = np.where((y_azimuth_test >= azimuth_range[0]) & (y_azimuth_test < azimuth_range[1]) &
                        (y_orientation_test >= orientation_range[0]) & (y_orientation_test < orientation_range[1]))[0]
        return X_test[find][:n_samples], y_orientation_test[find][:n_samples]



def cacl_RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def cacl_RMSE_wrapped_to_pi(y_true, y_pred):
    return np.sqrt(np.mean(wrapToPi(y_true - y_pred)**2))


def test_distance_custom_function(distance_model, X_test,
                                cost_function, y_distance_test, y_azimuth_test, y_orientation_test,
                                min_azimuth, max_azimuth,
                                min_orientation, max_orientation,
                                predict_fn, batch_size, device,
                                bins = 20, n_samples=100):
    # create list of azimuth and orientation ranges based on min max and bins
    azimuth_ranges = np.linspace(min_azimuth, max_azimuth, bins+1)
    orientation_ranges = np.linspace(min_orientation, max_orientation, bins+1)
    # create empty matrix to store the results
    results = np.zeros((bins, bins))
    # for each azimuth range
    for i, azimuth_range in enumerate(zip(azimuth_ranges[:-1], azimuth_ranges[1:])):
        # for each orientation range
        for j, orientation_range in enumerate(zip(orientation_ranges[:-1], orientation_ranges[1:])):
            # get the test data from the ranges
            X_test_, y_distance_test_ = get_test_distance_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                                                      azimuth_range, orientation_range, n_samples)
            # predict the distance
            y_distance_pred = predict_fn(distance_model, X_test_, batch_size, device)
            # calculate the cost function
            results[i,j] = cost_function(y_distance_test_, y_distance_pred)

    return results, azimuth_ranges, orientation_ranges

def test_distance_custom_function2(distance_model, X_test,
                                cost_function, y_distance_test, y_azimuth_test, y_orientation_test,
                                min_distance, max_distance,
                                min_orientation, max_orientation,
                                predict_fn, batch_size, device,
                                bins = 20, n_samples=100):
    # create list of azimuth and orientation ranges based on min max and bins
    distance_ranges = np.linspace(min_distance, max_distance, bins+1)
    orientation_ranges = np.linspace(min_orientation, max_orientation, bins+1)
    # create empty matrix to store the results
    results = np.zeros((bins, bins))
    # for each azimuth range
    for i, distance_range in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
        # for each orientation range
        for j, orientation_range in enumerate(zip(orientation_ranges[:-1], orientation_ranges[1:])):
            # get the test data from the ranges
            X_test_, y_distance_test_ = get_test_distance_from_ranges2(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                                                      distance_range, orientation_range, n_samples)

            # predict the distance
            y_distance_pred = predict_fn(distance_model, X_test_, batch_size, device)
            # calculate the cost function
            results[i,j] = cost_function(y_distance_test_, y_distance_pred)

    return results, distance_ranges, orientation_ranges

def test_distance_custom_function3(distance_model, X_test,
                                cost_function, y_distance_test, y_azimuth_test, y_orientation_test,
                                min_distance, max_distance,
                                min_azimuth, max_azimuth,
                                predict_fn, batch_size, device,
                                bins = 20, n_samples=100):
    # create list of azimuth and orientation ranges based on min max and bins
    distance_ranges = np.linspace(min_distance, max_distance, bins+1)
    azimuth_ranges = np.linspace(min_azimuth, max_azimuth, bins+1)
    # create empty matrix to store the results
    results = np.zeros((bins, bins))
    # for each azimuth range
    for i, distance_range in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
        # for each orientation range
        for j, azimuth_range in enumerate(zip(azimuth_ranges[:-1], azimuth_ranges[1:])):
            # get the test data from the ranges
            X_test_, y_distance_test_ = get_test_distance_from_ranges3(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                                                      distance_range, azimuth_range, n_samples)
            # predict the distance
            y_distance_pred = predict_fn(distance_model, X_test_, batch_size, device)
            # calculate the cost function
            results[i,j] = cost_function(y_distance_test_, y_distance_pred)
    return results, distance_ranges, azimuth_ranges


def test_azimuth_custom_function(azimuth_model, X_test,
                                cost_function, y_distance_test, y_azimuth_test, y_orientation_test,
                                min_distance, max_distance,
                                min_orientation, max_orientation,
                                predict_fn, batch_size, device,
                                bins = 20, n_samples=100):
    # create list of azimuth and orientation ranges based on min max and bins
    distance_ranges = np.linspace(min_distance, max_distance, bins+1)
    orientation_ranges = np.linspace(min_orientation, max_orientation, bins+1)
    # create empty matrix to store the results
    results = np.zeros((bins, bins))
    # for each azimuth range
    for i, distance_range in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
        # for each orientation range
        for j, orientation_range in enumerate(zip(orientation_ranges[:-1], orientation_ranges[1:])):
            # get the test data from the ranges
            X_test_, y_azimuth_test_ = get_test_azimuth_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                                                      distance_range, orientation_range, n_samples)
            # predict the distance
            y_azimuth_pred = predict_fn(azimuth_model, X_test_, batch_size, device)
            # calculate the cost function
            results[i,j] = cost_function(y_azimuth_test_, y_azimuth_pred)
    return results, distance_ranges, orientation_ranges


def test_azimuth_custom_function2(azimuth_model, X_test,
                                cost_function, y_distance_test, y_azimuth_test, y_orientation_test,
                                min_distance, max_distance,
                                min_azimuth, max_azimuth,
                                predict_fn, batch_size, device,
                                bins = 20, n_samples=100):
    # create list of azimuth and orientation ranges based on min max and bins
    distance_ranges = np.linspace(min_distance, max_distance, bins+1)
    azimuth_ranges = np.linspace(min_azimuth, max_azimuth, bins+1)
    # create empty matrix to store the results
    results = np.zeros((bins, bins))
    # for each azimuth range
    for i, distance_range in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
        # for each orientation range
        for j, azimuth_range in enumerate(zip(azimuth_ranges[:-1], azimuth_ranges[1:])):
            # get the test data from the ranges
            X_test_, y_azimuth_test_ = get_test_azimuth_from_ranges2(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                                                      distance_range, azimuth_range, n_samples)
            # predict the distance
            y_azimuth_pred = predict_fn(azimuth_model, X_test_, batch_size, device)
            # calculate the cost function
            results[i,j] = cost_function(y_azimuth_test_, y_azimuth_pred)

    return results, distance_ranges, azimuth_ranges

def test_azimuth_custom_function3(azimuth_model, X_test,
                                cost_function, y_distance_test, y_azimuth_test, y_orientation_test,
                                min_orientation, max_orientation,
                                min_azimuth, max_azimuth,
                                predict_fn, batch_size, device,
                                bins = 20, n_samples=100):
    # create list of azimuth and orientation ranges based on min max and bins
    orientation_ranges = np.linspace(min_orientation, max_orientation, bins+1)
    azimuth_ranges = np.linspace(min_azimuth, max_azimuth, bins+1)
    # create empty matrix to store the results
    results = np.zeros((bins, bins))
    # for each azimuth range
    for i, orientation_range in enumerate(zip(orientation_ranges[:-1], orientation_ranges[1:])):
        # for each orientation range
        for j, azimuth_range in enumerate(zip(azimuth_ranges[:-1], azimuth_ranges[1:])):
            # get the test data from the ranges
            X_test_, y_azimuth_test_ = get_test_azimuth_from_ranges3(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                                                      orientation_range, azimuth_range, n_samples)
            # predict the distance
            y_azimuth_pred = predict_fn(azimuth_model, X_test_, batch_size, device)
            # calculate the cost function
            results[i,j] = cost_function(y_azimuth_test_, y_azimuth_pred)

    return results, orientation_ranges, azimuth_ranges


def test_orientation_custom_function(orientation_model, X_test,
                                    cost_function, y_distance_test, y_azimuth_test, y_orientation_test,
                                    min_distance, max_distance,
                                    min_azimuth, max_azimuth,
                                    predict_fn, batch_size, device,
                                    bins = 20, n_samples=100):
    # create list of azimuth and orientation ranges based on min max and bins
    distance_ranges = np.linspace(min_distance, max_distance, bins+1)
    azimuth_ranges = np.linspace(min_azimuth, max_azimuth, bins+1)
    # create empty matrix to store the results
    results = np.zeros((bins, bins))
    # for each azimuth range
    for i, distance_range in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
        # for each orientation range
        for j, azimuth_range in enumerate(zip(azimuth_ranges[:-1], azimuth_ranges[1:])):
            # get the test data from the ranges
            X_test_, y_orientation_test_ = get_test_orientation_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                                                      distance_range, azimuth_range, n_samples)
            # predict the distance
            y_orientation_pred = predict_fn(orientation_model, X_test_, batch_size, device)
            # calculate the cost function
            results[i,j] = cost_function(y_orientation_test_, y_orientation_pred)
    return results, distance_ranges, azimuth_ranges

def test_orientation_custom_function2(orientation_model, X_test,
                                    cost_function, y_distance_test, y_azimuth_test, y_orientation_test,
                                    min_distance, max_distance,
                                    min_orientation, max_orientation,
                                    predict_fn, batch_size, device,
                                    bins = 20, n_samples=100):
    # create list of azimuth and orientation ranges based on min max and bins
    distance_ranges = np.linspace(min_distance, max_distance, bins+1)
    orientation_ranges = np.linspace(min_orientation, max_orientation, bins+1)
    # create empty matrix to store the results
    results = np.zeros((bins, bins))
    # for each azimuth range
    for i, distance_range in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
        # for each orientation range
        for j, orientation_range in enumerate(zip(orientation_ranges[:-1], orientation_ranges[1:])):
            # get the test data from the ranges
            X_test_, y_orientation_test_ = get_test_orientation_from_ranges2(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                                                      distance_range, orientation_range, n_samples)
            # predict the distance
            y_orientation_pred = predict_fn(orientation_model, X_test_, batch_size, device)
            # calculate the cost function
            results[i,j] = cost_function(y_orientation_test_, y_orientation_pred)
    return results, distance_ranges, orientation_ranges

def test_orientation_custom_function3(orientation_model, X_test,
                                    cost_function, y_distance_test, y_azimuth_test, y_orientation_test,
                                    min_azimuth, max_azimuth,
                                    min_orientation, max_orientation,
                                    predict_fn, batch_size, device,
                                    bins = 20, n_samples=100):
    # create list of azimuth and orientation ranges based on min max and bins
    azimuth_ranges = np.linspace(min_azimuth, max_azimuth, bins+1)
    orientation_ranges = np.linspace(min_orientation, max_orientation, bins+1)
    # create empty matrix to store the results
    results = np.zeros((bins, bins))
    # for each azimuth range
    for i, azimuth_range in enumerate(zip(azimuth_ranges[:-1], azimuth_ranges[1:])):
        # for each orientation range
        for j, orientation_range in enumerate(zip(orientation_ranges[:-1], orientation_ranges[1:])):
            # get the test data from the ranges
            X_test_, y_orientation_test_ = get_test_orientation_from_ranges3(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                                                      azimuth_range, orientation_range, n_samples)
            # predict the distance
            y_orientation_pred = predict_fn(orientation_model, X_test_, batch_size, device)
            # calculate the cost function
            results[i,j] = cost_function(y_orientation_test_, y_orientation_pred)
    return results, azimuth_ranges, orientation_ranges


def plot_results(results, axes, x_range, y_range, title, xlabel, ylabel, vmin=None, vmax=None):
    im = axes.imshow(results, origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
    
    axes.set_xticks([0, results.shape[0]/2-0.5, results.shape[0]-1])
    axes.set_yticks([0, results.shape[1]/2-0.5, results.shape[1]-1])
    axes.set_xticklabels(np.round(np.take(x_range, [0, int(len(x_range)/2), len(x_range)-1]), 1), fontsize=6)
    axes.set_yticklabels(np.round(np.take(y_range, [0, int(len(y_range)/2), len(y_range)-1]), 1), fontsize=6)
    #axes.set_xticks(np.arange(len(x_range)-1)[::5])
    #axes.set_yticks(np.arange(len(y_range)-1)[::5])
    #axes.set_xticklabels(np.round(x_range[:-1][::5], 1))
    #axes.set_yticklabels(np.round(y_range[:-1][::5], 1))
    axes.set_title(title, fontsize=6)
    axes.set_xlabel(xlabel, fontsize=6)
    axes.set_ylabel(ylabel, fontsize=6)
    axes.set_aspect('equal')
    return im
    

def run_custom_test_and_plot(distance_estimator, azimuth_estimator, orientation_estimator,
        X_test, y_distance_test, y_azimuth_test, y_orientation_test,
        min_distance=0.2, max_distance=3.2,
        min_azimuth=-np.radians(110), max_azimuth=np.radians(110),
        min_orientation=-np.pi, max_orientation=np.pi,):
    distance_results, azimuth_ranges, orientation_ranges = test_distance_custom_function(distance_estimator, X_test,
                                                                                         cacl_RMSE, y_distance_test,
                                                                                         min_azimuth, max_azimuth,
                                                                                         min_orientation, max_orientation,
                                                                                         bins = 10, n_samples=50)
    azimuth_results, distance_ranges, orientation_ranges = test_azimuth_custom_function(azimuth_estimator, X_test,
                                                                                            cacl_RMSE_wrapped_to_pi, y_azimuth_test,
                                                                                            min_distance, max_distance,
                                                                                            min_orientation, max_orientation,
                                                                                            bins = 10, n_samples=50)
    orientation_results, distance_ranges, azimuth_ranges = test_orientation_custom_function(orientation_estimator, X_test,
                                                                                                cacl_RMSE_wrapped_to_pi, y_orientation_test,
                                                                                                min_distance, max_distance,
                                                                                                min_azimuth, max_azimuth,
                                                                                                bins = 10, n_samples=50)
    fig, ax = plt.subplots(1, 3, dpi=200)

    im0 = plot_results(distance_results, ax[0], orientation_ranges, azimuth_ranges,'Distance', 'Orientation (rad)', 'Azimuth (rad)', vmin=0, vmax=0.6)
    im1 = plot_results(azimuth_results, ax[1], orientation_ranges, distance_ranges, 'Azimuth', 'Orientation (rad)', 'Distance (m)', vmin=0, vmax=np.radians(70))
    im2 = plot_results(orientation_results, ax[2], azimuth_ranges, distance_ranges, 'Orientation', 'Azimuth (rad)', 'Distance (m)', vmin=0, vmax=np.radians(120))
    # add color bar with set witdth and height
    fig.tight_layout()

    ims = [im0, im1, im2]
    
    return distance_results, azimuth_results, orientation_results, \
            fig, ax, ims


def run_custom_test_and_plot_9figset(distance_estimator, azimuth_estimator, orientation_estimator,
        X_test, y_distance_test, y_azimuth_test, y_orientation_test,
        min_distance=0.2, max_distance=3.2, 
        min_azimuth=-np.radians(110), max_azimuth=np.radians(110),
        min_orientation=-np.pi, max_orientation=np.pi):
    distance_results0, azimuth_ranges0, orientation_ranges0 = test_distance_custom_function(distance_estimator, X_test,
                                                                                            cacl_RMSE, y_distance_test,
                                                                                            min_azimuth, max_azimuth,
                                                                                            min_orientation, max_orientation,
                                                                                            bins = 10, n_samples=10)
    distance_results1, distance_ranges1, orientation_ranges1 = test_distance_custom_function2(distance_estimator, X_test,
                                                                                            cacl_RMSE, y_distance_test,
                                                                                            min_distance, max_distance,
                                                                                            min_orientation, max_orientation,
                                                                                            bins = 10, n_samples=10)
    distance_result2, distance_ranges2, azimuth_ranges2 = test_distance_custom_function3(distance_estimator, X_test,
                                                                                            cacl_RMSE, y_distance_test,
                                                                                            min_distance, max_distance,
                                                                                            min_azimuth, max_azimuth,
                                                                                            bins = 10, n_samples=10)
    azimuth_results0, distance_ranges3, orientation_ranges3 = test_azimuth_custom_function(azimuth_estimator, X_test,
                                                                                            cacl_RMSE_wrapped_to_pi, y_azimuth_test,
                                                                                            min_distance, max_distance,
                                                                                            min_orientation, max_orientation,
                                                                                            bins = 10, n_samples=10)
    azimuth_results1, distance_ranges4, azimuth_ranges4 = test_azimuth_custom_function2(azimuth_estimator, X_test,
                                                                                            cacl_RMSE_wrapped_to_pi, y_azimuth_test,
                                                                                            min_distance, max_distance,
                                                                                            min_azimuth, max_azimuth,
                                                                                            bins = 10, n_samples=10)
    azimuth_results2, orientation_ranges5, azimuth_ranges5 = test_azimuth_custom_function3(azimuth_estimator, X_test,
                                                                                            cacl_RMSE_wrapped_to_pi, y_azimuth_test,
                                                                                            min_orientation, max_orientation,
                                                                                            min_azimuth, max_azimuth,
                                                                                            bins = 10, n_samples=10)
    orientation_results0, distance_ranges6, azimuth_ranges6 = test_orientation_custom_function(orientation_estimator, X_test,
                                                                                                cacl_RMSE_wrapped_to_pi, y_orientation_test,
                                                                                                min_distance, max_distance,
                                                                                                min_azimuth, max_azimuth,
                                                                                                bins = 10, n_samples=10)
    orientation_results1, distance_ranges7, orientation_ranges7 = test_orientation_custom_function2(orientation_estimator, X_test,
                                                                                                cacl_RMSE_wrapped_to_pi, y_orientation_test,
                                                                                                min_distance, max_distance,
                                                                                                min_orientation, max_orientation,
                                                                                                bins = 10, n_samples=10)
    orientation_results2, azimuth_ranges8, orientation_ranges8 = test_orientation_custom_function3(orientation_estimator, X_test,
                                                                                                cacl_RMSE_wrapped_to_pi, y_orientation_test,
                                                                                                min_azimuth, max_azimuth,
                                                                                                min_orientation, max_orientation,
                                                                                                bins = 10, n_samples=10)

    fig, ax = plt.subplots(3, 3, dpi=200, figsize=(6,6))
    fig.subplots_adjust(right=0.9)
    im0 = plot_results(distance_results0, ax[0,0], orientation_ranges0, azimuth_ranges0,'Distance', 'Orientation (rad)', 'Azimuth (rad)', vmin=0, vmax=.8)
    im1 = plot_results(distance_results1, ax[0,1], orientation_ranges1, distance_ranges1, 'Distance', 'Orientation (rad)', 'Distance (m)', vmin=0, vmax=.8)
    im2 = plot_results(distance_result2, ax[0,2], azimuth_ranges2, distance_ranges2, 'Distance', 'Azimuth (rad)', 'Distance (m)', vmin=0, vmax=.8)
    im3 = plot_results(azimuth_results0, ax[1,0], orientation_ranges3, distance_ranges3, 'Azimuth', 'Orientation (rad)', 'Distance (m)', vmin=0, vmax=np.radians(80))
    im4 = plot_results(azimuth_results1, ax[1,1], azimuth_ranges4, distance_ranges4, 'Azimuth', 'Azimuth (rad)', 'Distance (m)', vmin=0, vmax=np.radians(80))
    im5 = plot_results(azimuth_results2, ax[1,2], azimuth_ranges5, orientation_ranges5, 'Azimuth', 'Azimuth (rad)', 'Orientation (rad)', vmin=0, vmax=np.radians(80))
    im6 = plot_results(orientation_results0, ax[2,0], azimuth_ranges6, distance_ranges6, 'Orientation', 'Azimuth (rad)', 'Distance (m)', vmin=0, vmax=np.radians(150))
    im7 = plot_results(orientation_results1, ax[2,1], orientation_ranges7, distance_ranges7, 'Orientation', 'Orientation (rad)','Distance (m)',  vmin=0, vmax=np.radians(150))
    im8 = plot_results(orientation_results2, ax[2,2], orientation_ranges8, azimuth_ranges8, 'Orientation', 'Orientation (rad)', 'Azimuth (rad)', vmin=0, vmax=np.radians(150))
    fig.colorbar(im2, cax=fig.add_axes([0.99, 0.75, 0.01, 0.2]))
    fig.colorbar(im5, cax=fig.add_axes([0.99, 0.42, 0.01, 0.2]))
    fig.colorbar(im8, cax=fig.add_axes([0.99, 0.1, 0.01, 0.2]))
    # add color bar with set witdth and height
    fig.tight_layout()

    ims = [im0, im1, im2, im3, im4, im5, im6, im7, im8]

    return distance_results0, distance_results1, distance_result2, \
        azimuth_results0, azimuth_results1, azimuth_results2, \
            orientation_results0, orientation_results1, orientation_results2, \
                fig, ax, ims
