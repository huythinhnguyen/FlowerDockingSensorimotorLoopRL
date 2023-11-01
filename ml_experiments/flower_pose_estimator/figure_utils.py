import numpy as np

from Sensors.FlowerEchoSimulator.Spatializer import wrapToPi
from sklearn.utils import shuffle
from .eval_utils import predict

from matplotlib import pyplot as plt

def get_test_data_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                              distance_range, azimuth_range, orientation_range, n_samples=100, shuffle=False):
    find = np.where(
        (y_distance_test >= distance_range[0]) & (y_distance_test < distance_range[1]) &
        (y_azimuth_test >= azimuth_range[0]) & (y_azimuth_test < azimuth_range[1]) &
        (y_orientation_test >= orientation_range[0]) & (y_orientation_test < orientation_range[1])
    )[0]

    return X_test[find][:n_samples], y_distance_test[find][:n_samples], \
        y_azimuth_test[find][:n_samples], y_orientation_test[find][:n_samples]

def cacl_RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def cacl_RMSE_wrapped_to_pi(y_true, y_pred):
    return np.sqrt(np.mean(wrapToPi(y_true - y_pred)**2))
# test_model_on_azimuth_orientation_grid
def test_model_on_azimuth_orientation_grid(model, X_test,
                                    y_distance_test, y_azimuth_test, y_orientation_test,
                                    min_distance, max_distance,
                                    min_azimuth, max_azimuth,
                                    min_orientation, max_orientation,
                                    batchsize, device, bins=10,
                                    transform=None):
    azimuth_ranges = np.linspace(min_azimuth, max_azimuth, bins+1)
    orientation_ranges = np.linspace(min_orientation, max_orientation, bins+1)

    distance_results = np.zeros((bins, bins))
    azimuth_results = np.zeros((bins, bins))
    orientation_results = np.zeros((bins, bins))
    
    for i, orientation_range in enumerate(zip(orientation_ranges[:-1], orientation_ranges[1:])):
        for j, azimuth_range in enumerate(zip(azimuth_ranges[:-1], azimuth_ranges[1:])):
            X_test_cell, y_distance_test_cell, y_azimuth_test_cell, y_orientation_test_cell \
                = get_test_data_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                            distance_range=(min_distance, max_distance),
                                            azimuth_range=azimuth_range,
                                            orientation_range=orientation_range,)                                            

            y_pred = predict(model, X_test_cell, batch_size=batchsize, device=device, transform=transform)
            
            distance_results[i,j] = cacl_RMSE(y_distance_test_cell, y_pred[0])
            azimuth_results[i,j] = cacl_RMSE_wrapped_to_pi(y_azimuth_test_cell, y_pred[1])
            orientation_results[i,j] = cacl_RMSE_wrapped_to_pi(y_orientation_test_cell, y_pred[2])

    return distance_results, azimuth_results, orientation_results

def test_model_on_distance_orientation_grid(model, X_test,
                                        y_distance_test, y_azimuth_test, y_orientation_test,
                                        min_distance, max_distance,
                                        min_azimuth, max_azimuth,
                                        min_orientation, max_orientation,
                                        batchsize, device, bins=10,
                                        transform=None):
    distance_ranges = np.linspace(min_distance, max_distance, bins+1)
    orientation_ranges = np.linspace(min_orientation, max_orientation, bins+1)

    distance_results = np.zeros((bins, bins))
    azimuth_results = np.zeros((bins, bins))
    orientation_results = np.zeros((bins, bins))

    for i, distance_range in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
        for j, orientation_range in enumerate(zip(orientation_ranges[:-1], orientation_ranges[1:])):
            X_test_cell, y_distance_test_cell, y_azimuth_test_cell, y_orientation_test_cell \
                = get_test_data_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                            distance_range=distance_range,
                                            azimuth_range=(min_azimuth, max_azimuth),
                                            orientation_range=orientation_range,)
            y_pred = predict(model, X_test_cell, batchsize, device, transform=transform)
            distance_results[i,j] = cacl_RMSE(y_distance_test_cell, y_pred[0])
            azimuth_results[i,j] = cacl_RMSE_wrapped_to_pi(y_azimuth_test_cell, y_pred[1])
            orientation_results[i,j] = cacl_RMSE_wrapped_to_pi(y_orientation_test_cell, y_pred[2])

    return distance_results, azimuth_results, orientation_results

def test_model_on_distance_azimuth_grid(model, X_test,
                                        y_distance_test, y_azimuth_test, y_orientation_test,
                                        min_distance, max_distance,
                                        min_azimuth, max_azimuth,
                                        min_orientation, max_orientation,
                                        batchsize, device, bins=10,
                                        transform=None):
    distance_ranges = np.linspace(min_distance, max_distance, bins+1)
    azimuth_ranges = np.linspace(min_azimuth, max_azimuth, bins+1)

    distance_results = np.zeros((bins, bins))
    azimuth_results = np.zeros((bins, bins))
    orientation_results = np.zeros((bins, bins))

    for i, distance_range in enumerate(zip(distance_ranges[:-1], distance_ranges[1:])):
        for j, azimuth_range in enumerate(zip(azimuth_ranges[:-1], azimuth_ranges[1:])):
            X_test_cell, y_distance_test_cell, y_azimuth_test_cell, y_orientation_test_cell \
                = get_test_data_from_ranges(X_test, y_distance_test, y_azimuth_test, y_orientation_test,
                                            distance_range=distance_range,
                                            azimuth_range=azimuth_range,
                                            orientation_range=(min_orientation, max_orientation),)

            y_pred = predict(model, X_test_cell, batchsize, device, transform=transform)
            distance_results[i,j] = cacl_RMSE(y_distance_test_cell, y_pred[0])
            azimuth_results[i,j] = cacl_RMSE_wrapped_to_pi(y_azimuth_test_cell, y_pred[1])
            orientation_results[i,j] = cacl_RMSE_wrapped_to_pi(y_orientation_test_cell, y_pred[2])

    return distance_results, azimuth_results, orientation_results



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
    

def run_custom_test_and_plot_9figset(model, X_test,
                                    y_distance_test, y_azimuth_test, y_orientation_test,
                                    min_distance=0.2, max_distance=3.2,
                                    min_azimuth=-np.radians(110), max_azimuth=np.radians(110),
                                    min_orientation=-np.pi, max_orientation=np.pi, bins=10,
                                    max_dist_err=0.5, max_azimuth_err=np.radians(20), max_orientation_err=np.radians(90),
                                    transform=None):
    distance_results0, azimuth_results2, orientation_results2 \
        = test_model_on_azimuth_orientation_grid(model, X_test,
                                    y_distance_test, y_azimuth_test, y_orientation_test,
                                    min_distance=min_distance, max_distance=max_distance,
                                    min_azimuth=min_azimuth, max_azimuth=max_azimuth,
                                    min_orientation=min_orientation, max_orientation=max_orientation, bins=bins,
                                    transform=transform)
    
    distance_results1, azimuth_results0, orientation_results1 \
        = test_model_on_distance_orientation_grid(model, X_test,
                                    y_distance_test, y_azimuth_test, y_orientation_test,
                                    min_distance=min_distance, max_distance=max_distance,
                                    min_azimuth=min_azimuth, max_azimuth=max_azimuth,
                                    min_orientation=min_orientation, max_orientation=max_orientation, bins=bins,
                                    transform=transform)
    
    distance_results2, azimuth_results1, orientation_results0 \
        = test_model_on_distance_azimuth_grid(model, X_test,
                                    y_distance_test, y_azimuth_test, y_orientation_test,
                                    min_distance=min_distance, max_distance=max_distance,
                                    min_azimuth=min_azimuth, max_azimuth=max_azimuth,
                                    min_orientation=min_orientation, max_orientation=max_orientation, bins=bins,
                                    transform=transform)
    
    distance_ranges = np.linspace(min_distance, max_distance, bins+1)
    azimuth_ranges = np.linspace(min_azimuth, max_azimuth, bins+1)
    orientation_ranges = np.linspace(min_orientation, max_orientation, bins+1)

    fig, ax = plt.subplots(3, 3, figsize=(6,6), dpi=200)
    im0 = plot_results(distance_results0, ax[0,0], azimuth_ranges, orientation_ranges,
                       'Dist. RMSE.', 'Azimuth (rad)', 'Orientation (rad)', vmin=0, vmax=max_dist_err)
    im1 = plot_results(distance_results1, ax[0,1], orientation_ranges, distance_ranges,
                       'Dist. RMSE.', 'Orientation (rad)', 'Distance (m)', vmin=0, vmax=max_dist_err)
    im2 = plot_results(distance_results2, ax[0,2], azimuth_ranges, distance_ranges,
                       'Dist. RMSE.', 'Azimuth (rad)', 'Distance (m)', vmin=0, vmax=max_dist_err)
    im3 = plot_results(azimuth_results0, ax[1,0], orientation_ranges, distance_ranges,
                       'Azi. RMSE', 'Orientation (rad)', 'Distance (m)', vmin=0, vmax=max_azimuth_err)
    im4 = plot_results(azimuth_results1, ax[1,1], azimuth_ranges, distance_ranges,
                       'Azi. RMSE', 'Azimuth (rad)', 'Distance (m)', vmin=0, vmax=max_azimuth_err)
    im5 = plot_results(azimuth_results2, ax[1,2], azimuth_ranges, orientation_ranges,
                       'Azi. RMSE', 'Azimuth (rad)', 'Orientation (rad)', vmin=0, vmax=max_azimuth_err)
    im6 = plot_results(orientation_results0, ax[2,0], azimuth_ranges, distance_ranges,
                       'Ori. RMSE', 'Azimuth (rad)', 'Distance (m)', vmin=0, vmax=max_orientation_err)
    im7 = plot_results(orientation_results1, ax[2,1], orientation_ranges, distance_ranges,
                       'Ori. RSME', 'Orientation (rad)','Distance (m)',  vmin=0, vmax=max_orientation_err)
    im8 = plot_results(orientation_results2, ax[2,2], azimuth_ranges, orientation_ranges, 
                       'Ori. RSME', 'Azimuth (rad)', 'Orientation (rad)', vmin=0, vmax=max_orientation_err)
    fig.colorbar(im2, cax=fig.add_axes([0.99, 0.75, 0.01, 0.2]))
    fig.colorbar(im5, cax=fig.add_axes([0.99, 0.42, 0.01, 0.2]))
    fig.colorbar(im8, cax=fig.add_axes([0.99, 0.1, 0.01, 0.2]))
    fig.tight_layout()
    plt.show()

    return distance_results0, distance_results1, distance_results2, azimuth_results0, azimuth_results1, azimuth_results2, orientation_results0, orientation_results1, orientation_results2