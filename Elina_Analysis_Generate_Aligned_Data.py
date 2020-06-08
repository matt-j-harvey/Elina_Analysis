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
from sklearn.cluster import DBSCAN, AffinityPropagation, OPTICS, Birch, SpectralClustering
import community
from mpl_toolkits.mplot3d import Axes3D
import math

# Load Activity Matrix From CSV File
def load_activity_matrix(activity_file_location):
    print("loading activity matrix")
    activity_matrix = np.genfromtxt(activity_file_location, delimiter=",", dtype="float")
    return activity_matrix


# Load ROI Coordinates From CSV File
def load_roi_coordinates(roi_coorrdinates_file):
    print("Loading ROI Coordinates")
    roi_coordinates = np.genfromtxt(roi_coorrdinates_file, delimiter=",", dtype="float")
    return roi_coordinates


# Assign a colour based upon an activity value
def get_colour(input_value, colour_map, scale_factor):
    input_value = input_value * scale_factor
    cmap = cm.get_cmap(colour_map)
    float_tuple = cmap(input_value)
    matplot_to_q_colour_conversion_factor = 255
    colour = QColor(float_tuple[0] * matplot_to_q_colour_conversion_factor,
                    float_tuple[1] * matplot_to_q_colour_conversion_factor,
                    float_tuple[2] * matplot_to_q_colour_conversion_factor)

    return colour



def plot_all_tectums():

    list_of_all_fish = controls + cortisol_low + cortisol_high

    figure_1 = plt.figure()
    subplots = []

    for fish in range(number_of_fish):
        subplots.append(figure_1.add_subplot(3,3,fish+1))
        tectal_rois = np.load(base_directory + list_of_all_fish[fish] + "/Registered_Coords.npy")
        subplots[fish].scatter(tectal_rois[:, 0], tectal_rois[:, 1])

    plt.show()

    for fish in range(number_of_fish):
        tectal_rois = np.load(base_directory + list_of_all_fish[fish] + "/Registered_Coords.npy")
        plt.scatter(tectal_rois[:, 0], tectal_rois[:, 1], alpha=0.2)

    plt.show()


def get_spatial_patterns_of_activity(activity_matrix, sensor_coupling_matrix, sensor_density):

    number_of_neurons = np.shape(activity_matrix)[0]
    number_of_timepoints = np.shape(activity_matrix)[1]
    number_of_sensors = np.shape(sensor_coupling_matrix)[1]

    smoothed_matrix = np.zeros((number_of_timepoints, number_of_sensors))


    #plt.ion()
    for timepoint in range(0,number_of_timepoints):

        sensor_activity = np.matmul(activity_matrix[:, timepoint], sensor_coupling_matrix)
        smoothed_matrix[timepoint] = sensor_activity

        # print("timepoint", timepoint)
        #sensor_activity = np.reshape(sensor_activity, (sensor_density, sensor_density))
        #plt.title(timepoint)
        #plt.imshow(sensor_activity, cmap="jet", vmin=0, vmax=0.5) # vmin=0, vmax=0.2
        #plt.draw()
        #plt.pause(0.01)
        #plt.clf()

    return smoothed_matrix



def create_sensor_coupling_matrix(tectal_rois, sensor_density, save=False, save_location=None):

    #Create Sensor Centroids
    x_range = 300
    y_range = 300
    number_of_sensors = sensor_density * sensor_density
    sensor_x_spacing = x_range / sensor_density
    sensor_y_spacing = y_range / sensor_density

    centroids = []
    for x in range(sensor_density):
        for y in range(sensor_density):
            x_pos = x * sensor_x_spacing
            y_pos = y * sensor_y_spacing
            centroids.append([x_pos, y_pos])

    centroids = np.array(centroids)


    #Create Coupling Matrix
    gaussian_sd = 10
    number_of_neurons = np.shape(tectal_rois)[0]
    coupling_matrix = np.zeros((number_of_neurons, number_of_sensors))

    for neuron in range(number_of_neurons):
        #print(neuron)
        neuron_x = tectal_rois[neuron, 0]
        neuron_y = tectal_rois[neuron, 1]

        for sensor in range(number_of_sensors):
            sensor_x = centroids[sensor, 0]
            sensor_y = centroids[sensor, 1]

            x_dist = (neuron_x - sensor_x) ** 2
            y_dist = (neuron_y - sensor_y) ** 2
            distance = np.sqrt(x_dist + y_dist)

            coupling = (1 / (gaussian_sd * math.sqrt(2 * math.pi))) * math.e ** (-0.5 * (distance / gaussian_sd)**2)
            coupling_matrix[neuron, sensor] = coupling

    return coupling_matrix



controls = ["/Controls/Fish_1", "/Controls/Fish_2"]
cortisol_high = ["/100uM/Fish_1", "/100uM/Fish_2", "/100uM/Fish_3"]
cortisol_low = ["40uM/Fish_1", "40uM/Fish_2", "40uM/Fish_3"]

all_fish = controls + cortisol_low + cortisol_high
number_of_fish = len(all_fish)

base_directory                  = "/home/matthew/Documents/Elina_Data/"
activity_matrix_file            = "/Tectum Binary Matrix.csv"
normalised_activity_file_pre    = "/Normalised_Activity_Matrix_Pre.npy"
normalised_activity_file_post   = "/Normalised_Activity_Matrix_Post.npy"

widefield_matrix_pre_file       = "/Widefield_Matrix_Pre"
widefield_matrix_post_file      = "/Widefield_Matrix_Post"
sensor_coupling_file            = "/Sensor_Coupling_Matrix.npy"


for fish in all_fish:
    print(fish)

    tectal_rois = np.load(base_directory + fish + "/Registered_Coords.npy")

    pre_activity_matrix = np.load(base_directory + fish + normalised_activity_file_pre)
    post_activity_matrix = np.load(base_directory + fish + normalised_activity_file_post)

    sensor_density = 100
    sensor_coupling_matrix = create_sensor_coupling_matrix(tectal_rois, sensor_density, save=True, save_location=sensor_coupling_file)
    np.save(base_directory + fish + sensor_coupling_file, sensor_coupling_matrix)

    widefield_matrix_pre  = get_spatial_patterns_of_activity(pre_activity_matrix, sensor_coupling_matrix, sensor_density)
    widefield_matrix_post = get_spatial_patterns_of_activity(post_activity_matrix, sensor_coupling_matrix, sensor_density)



    np.save(base_directory + fish + widefield_matrix_pre_file,  widefield_matrix_pre)
    np.save(base_directory + fish + widefield_matrix_post_file, widefield_matrix_post)




