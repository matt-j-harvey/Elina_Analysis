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

# Smooth With Moving Average
def smooth_traces(activity_matrix):
    print("Smoothing Trace")
    normalised_activity_matrix = []
    number_of_neurons = np.shape(activity_matrix)[0]
    sliding_window_size = 5

    for neuron in range(number_of_neurons):
        trace = np.clip(activity_matrix[neuron], a_min=0, a_max=None)
        normalised_trace = np.convolve(trace, np.ones((sliding_window_size,)) / sliding_window_size, mode='valid')
        normalised_trace = np.divide(normalised_trace, max(trace))
        normalised_activity_matrix.append(normalised_trace)

    normalised_activity_matrix = np.array(normalised_activity_matrix)
    return normalised_activity_matrix


# Scale So Max is 1 and Min is 0
def scale_traces(activity_matrix):

    number_of_neurons = np.shape(activity_matrix)[0]

    for neuron in range(number_of_neurons):

        neuron_activity = activity_matrix[neuron]

        #Shift Baseline So All Have A Minimum of 0
        neuron_minimum = np.min(neuron_activity)
        if neuron_minimum < 0:
            neuron_activity = np.add(neuron_activity, abs(neuron_minimum))
        elif neuron_minimum > 0:
            neuron_activity = np.subtract(neuron_activity, neuron_minimum)

        #Scale Activity So All Have a Maximum of 1
        neuron_maximum = np.max(neuron_activity)
        neuron_activity = np.divide(neuron_activity, neuron_maximum)


        activity_matrix[neuron] = neuron_activity

    return activity_matrix



base_directory                = "/home/matthew/Documents/Elina_Data/"
activity_matrix_file          = "/Tectum Binary Matrix.csv"
normalised_activity_file_pre  =  "/Normalised_Activity_Matrix_Pre.npy"
normalised_activity_file_post =  "/Normalised_Activity_Matrix_Post.npy"

controls      = ["/Controls/Fish_1", "/Controls/Fish_2"]
cortisol_high = ["/100uM/Fish_1", "/100uM/Fish_2", "/100uM/Fish_3"]
cortisol_low  = ["40uM/Fish_1", "40uM/Fish_2", "40uM/Fish_3"]

all_fish = controls + cortisol_low + cortisol_high
number_of_fish = len(all_fish)

recording_length = 34918
discard_time = 5820


for fish in all_fish:
    print(fish)

    activity_matrix = load_activity_matrix(base_directory + fish + activity_matrix_file)
    print("Activity Matrix Shape", np.shape(activity_matrix))

    # Normalsie Activity
    activity_matrix = smooth_traces(activity_matrix)
    activity_matrix = scale_traces(activity_matrix)

    #Split Into Pre and Post and Save
    pre_activity  = activity_matrix[:, discard_time:recording_length]
    post_activity = activity_matrix[:, recording_length+discard_time:]

    print("Pre activity shape", np.shape(pre_activity))
    print("Post activity shape", np.shape(post_activity))

    np.save(base_directory + fish + normalised_activity_file_pre,  pre_activity)
    np.save(base_directory + fish + normalised_activity_file_post, post_activity)
