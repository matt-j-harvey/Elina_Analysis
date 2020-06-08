import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import stats
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QBrush, QColor, QPixmap, QPainter
from PyQt5.QtCore import Qt, QLineF
import sys
from time import sleep
from scipy import ndimage, stats
from sklearn.decomposition import PCA, FactorAnalysis, NMF
from sklearn.cluster import DBSCAN, AffinityPropagation, OPTICS, Birch, SpectralClustering, MiniBatchKMeans
import community
from mpl_toolkits.mplot3d import Axes3D
import math


def get_combined_activity_matrix():
    print("Getting combined matrix")

    #GetFish 1
    widefield_matrix_pre  = np.load(base_directory + all_fish[0] + "/Widefield_Matrix_Pre.npy")
    widefield_matrix_post = np.load(base_directory + all_fish[0] + "/Widefield_Matrix_Post.npy")
    combined_matrix = np.vstack((widefield_matrix_pre,widefield_matrix_post))

    for fish in all_fish[1:]:
        widefield_activity = np.load(base_directory + fish + "/Widefield_Matrix_Pre.npy")
        combined_matrix = np.vstack((combined_matrix, widefield_activity))

        widefield_activity = np.load(base_directory + fish + "/Widefield_Matrix_Post.npy")
        combined_matrix = np.vstack((combined_matrix, widefield_activity))

    return combined_matrix



def factorise_matrix(combined_matrix):
    print("Facotrising Marix")
    print("Combined matrix shape", np.shape(combined_matrix))

    number_of_components = 20
    model = FactorAnalysis(n_components=number_of_components).fit(combined_matrix)
    components = model.components_
    transformed_points = model.transform(combined_matrix)

    return components, transformed_points



def factor_number(number_to_factor):
    factor_list = []

    for potential_factor in range(1, number_to_factor):
        if number_to_factor % potential_factor == 0:
            factor_pair = [potential_factor, int(number_to_factor/potential_factor)]
            factor_list.append(factor_pair)

    return factor_list



def get_best_grid(number_of_items):
    factors = factor_number(number_of_items)
    factor_difference_list = []

    #Get Difference Between All Factors
    for factor_pair in factors:
        factor_difference = abs(factor_pair[0] - factor_pair[1])
        factor_difference_list.append(factor_difference)

    #Select Smallest Factor difference
    smallest_difference = np.min(factor_difference_list)
    best_pair = factor_difference_list.index(smallest_difference)

    return factors[best_pair]


def display_components():

    components = np.load(base_directory + "/Combined_Components.npy")

    number_of_components = np.shape(components)[0]
    grid_dimensions = get_best_grid(number_of_components)

    figure_1 = plt.figure()


    axes = []

    selected_component = 0
    for y in range(grid_dimensions[0]):
        for x in range(grid_dimensions[1]):
            component_data = components[selected_component]

            max_value = np.max(component_data)
            min_value = abs(np.min(component_data))
            image_range = np.max([max_value, min_value])
            image = np.ndarray.reshape(component_data, (100,100))
            axes.append([])
            axes[selected_component] = figure_1.add_subplot(grid_dimensions[0], grid_dimensions[1], selected_component+1)

            axes[selected_component].imshow(image, cmap='jet', vmax=image_range, vmin=0)#-image_range
            axes[selected_component].axis('off')
            axes[selected_component].set_title(str(selected_component+1))

            selected_component += 1

    plt.tight_layout()
    plt.savefig(base_directory + "/Factors.png")
    plt.show()




def get_moving_average(trace, window_size):

    average_trace = []
    number_of_timepoints = np.shape(trace)[0]

    for timepoint in range(number_of_timepoints-window_size):

        stop_point = timepoint + window_size

        """
        if stop_point >= number_of_timepoints:
            stop_point = number_of_timepoints-1
        """
        average = np.mean(trace[timepoint:stop_point])
        average_trace.append(average)

    average_trace = np.array(average_trace)

    return average_trace


def scale_trace(neuron_activity):

    # Shift Baseline So All Have A Minimum of 0
    neuron_minimum = np.min(neuron_activity)
    if neuron_minimum < 0:
        neuron_activity = np.add(neuron_activity, abs(neuron_minimum))
    elif neuron_minimum > 0:
        neuron_activity = np.subtract(neuron_activity, neuron_minimum)

    # Scale Activity So All Have a Maximum of 1
    neuron_maximum = np.max(neuron_activity)
    neuron_activity = np.divide(neuron_activity, neuron_maximum)
    return neuron_activity


def plot_trajectories():

    transformed_points = np.load(base_directory + "/Transformed_points.npy")
    number_of_factors = np.shape(transformed_points)[1]

    for factor in range(number_of_factors):
        plot_factor_trajectory(transformed_points, factor)



def plot_factor_trajectory(transformed_points, factor):


    recording_length = 29098
    colourmap = cm.get_cmap("hsv")

    for fish in range(number_of_fish):
        pre_start = fish * recording_length * 2
        pre_end = pre_start + recording_length
        post_start = pre_end
        post_end = post_start + recording_length

        trajectory = transformed_points[pre_start:post_end, factor]
        trajectory = get_moving_average(trajectory, 5000)
        trajectory = scale_trace(trajectory)

        condition = fish_conditions[fish]
        colour = colourmap(float(condition)/3)
        plt.title(str(factor + 1))
        plt.plot(trajectory, c=colour)

    plt.vlines(recording_length, ymin=0, ymax=1)
    plt.savefig(images_directory + str(factor+1) + ".png")
    plt.close()



controls = ["/Controls/Fish_1", "/Controls/Fish_2"]
cortisol_high = ["/100uM/Fish_1", "/100uM/Fish_2", "/100uM/Fish_3"]
cortisol_low = ["40uM/Fish_1", "40uM/Fish_2", "40uM/Fish_3"]
all_fish = controls + cortisol_low + cortisol_high
fish_conditions = [1, 1, 2, 2, 2, 3, 3, 3]
number_of_fish = len(all_fish)

base_directory             = "/home/matthew/Documents/Elina_Data/"
widefield_matrix_pre_file  = "/Widefield_Matrix_Pre"
widefield_matrix_post_file = "/Widefield_Matrix_Post"

images_directory = base_directory + "/Loading_Traces/"

"""
combined_matrix = get_combined_activity_matrix()
components, transformed_points = factorise_matrix(combined_matrix)
np.save(base_directory + "/Combined_Components.npy", components)
np.save(base_directory + "/Transformed_points.npy",  transformed_points)
"""


display_components()
plot_trajectories()
