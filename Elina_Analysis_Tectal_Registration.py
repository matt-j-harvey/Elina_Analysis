from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import imageio
import numpy as np
from scipy import ndimage
import skimage.transform
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm
import PIL.Image
from skimage.transform import resize
from scipy import stats
import math


class tectum_matching_window(QWidget):

    def __init__(self, parent=None):
        super(tectum_matching_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Tectum Registration")
        self.setGeometry(0, 0, 1000, 500)
        self.show()

        # Create Figures
        self.display_figure = Figure()
        self.display_canvas = FigureCanvas(self.display_figure)
        self.display_axis   = self.display_figure.add_subplot(1,1,1)

        # Create Buttons
        self.left_button = QPushButton("Left")
        self.left_button.clicked.connect(self.move_left)

        self.right_button = QPushButton("Right")
        self.right_button.clicked.connect(self.move_right)

        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(self.move_up)

        self.down_button = QPushButton("Down")
        self.down_button.clicked.connect(self.move_down)

        self.rotate_clockwise_button = QPushButton("Rotate Clockwise")
        self.rotate_clockwise_button.clicked.connect(self.rotate_clockwise)

        self.rotate_counterclockwise_button = QPushButton("Rotate Counterclockwise")
        self.rotate_counterclockwise_button.clicked.connect(self.rotate_counterclockwise)

        self.enlarge_button = QPushButton("Enlarge")
        self.enlarge_button.clicked.connect(self.enlarge)

        self.shrink_button = QPushButton("Shrink")
        self.shrink_button.clicked.connect(self.shrink)

        self.map_button = QPushButton("Map Regions")
        self.map_button.clicked.connect(save_points)

        # Add Labels
        self.x_label = QLabel()
        self.y_label = QLabel()
        self.height_label = QLabel()
        self.width_label = QLabel()
        self.angle_label = QLabel()

        """
        self.y_label.setText("y: " + str(template_y))
        self.x_label.setText("x: " + str(template_x))
        self.width_label.setText("Width: " + str(bounding_width))
        self.height_label.setText("Height: " + str(bounding_height))
        self.angle_label.setText("Angle: " + str(rotation))
        """
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.display_canvas, 0, 0, 13, 8)

        self.layout.addWidget(self.left_button, 0, 16, 1, 1)
        self.layout.addWidget(self.right_button, 1, 16, 1, 1)
        self.layout.addWidget(self.up_button, 2, 16, 1, 1)
        self.layout.addWidget(self.down_button, 3, 16, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button, 4, 16, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button, 5, 16, 1, 1)
        self.layout.addWidget(self.enlarge_button, 6, 16, 1, 1)
        self.layout.addWidget(self.shrink_button, 7, 16, 1, 1)

        self.layout.addWidget(self.x_label, 8, 16, 1, 1)
        self.layout.addWidget(self.y_label, 9, 16, 1, 1)
        self.layout.addWidget(self.height_label, 10, 16, 1, 1)
        self.layout.addWidget(self.width_label, 11, 16, 1, 1)
        self.layout.addWidget(self.angle_label, 12, 16, 1, 1)

        self.layout.addWidget(self.map_button, 13, 16, 1, 1)


    def move_left(self):
        global x_shift
        x_shift += 2
        self.x_label.setText("x: " + str(x_shift))
        draw_images()

    def move_right(self):
        global x_shift
        x_shift -= 2
        self.x_label.setText("x: " + str(x_shift))
        draw_images()
        #draw_image(self.anatomy_figure, self.anatomy_canvas, self.activity_figure, self.activity_canvas)

    def move_up(self):
        global y_shift
        y_shift -= 2
        self.y_label.setText("x: " + str(y_shift))
        draw_images()

    def move_down(self):
        global y_shift
        y_shift += 2
        self.y_label.setText("x: " + str(y_shift))
        draw_images()

    def rotate_clockwise(self):
        global rotation
        rotation -= 1
        self.angle_label.setText("Angle: " + str(rotation))
        draw_images()


    def rotate_counterclockwise(self):
        global rotation
        rotation += 1
        self.angle_label.setText("Angle: " + str(rotation))
        draw_images()


    def enlarge(self):
        global scaling_factor
        scaling_factor += 0.05

        self.width_label.setText("Width: " + str(scaling_factor))
        self.height_label.setText("Height: " + str(scaling_factor))
        draw_images()
        #draw_image(self.anatomy_figure, self.anatomy_canvas, self.activity_figure, self.activity_canvas)

    def shrink(self):
        global scaling_factor
        scaling_factor -= 0.05

        self.width_label.setText("Width: " + str(scaling_factor))
        self.height_label.setText("Height: " + str(scaling_factor))
        draw_images()

        #draw_image(self.anatomy_figure, self.anatomy_canvas, self.activity_figure, self.activity_canvas)


# Load ROI Coordinates From CSV File
def load_roi_coordinates(roi_coorrdinates_file):
    print("Loading ROI Coordinates")
    roi_coordinates = np.genfromtxt(roi_coorrdinates_file, delimiter=",", dtype="float")
    return roi_coordinates



def rotate(origin, point, angle):

    angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate_points(points_to_match):
    new_coords = []
    number_of_neurons = np.shape(points_to_match)[0]

    for neuron in range(number_of_neurons):
        coordinates = points_to_match[neuron, 0:2]
        rotated_coords = rotate([250,250], coordinates, rotation)
        print(coordinates)
        print(rotated_coords)
        new_coords.append(rotated_coords)

    new_coords = np.array(new_coords)
    return new_coords


def draw_images():

    transformed_coords = np.copy(coords_to_match)

    transformed_coords = rotate_points(transformed_coords)

    transformed_coords[:, 0] = np.add(transformed_coords[:, 0],  x_shift)
    transformed_coords[:, 1] = np.add(transformed_coords[:, 1], y_shift)
    transformed_coords = np.multiply(transformed_coords, scaling_factor)

    window_instance.display_axis.clear()
    window_instance.display_axis.imshow(combined_matrix)
    window_instance.display_axis.scatter(template_coords[:, 0], template_coords[:, 1], c='b')
    window_instance.display_axis.scatter(transformed_coords[:, 0], transformed_coords[:, 1], c='g', alpha=0.3)

    window_instance.display_canvas.draw()
    window_instance.display_canvas.update()
    app.processEvents()


def save_points():

    transformed_coords = np.copy(coords_to_match)
    transformed_coords = rotate_points(transformed_coords)

    transformed_coords[:, 0] = np.add(transformed_coords[:, 0],  x_shift)
    transformed_coords[:, 1] = np.add(transformed_coords[:, 1], y_shift)
    transformed_coords = np.multiply(transformed_coords, scaling_factor)

    np.save(base_directory + fish_in_question + "/Registered_Coords.npy", transformed_coords)
    print("Saved!")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    global template_points
    global points_to_match
    global scaling_factor
    global x_shift
    global y_shift
    global rotation
    global transformed_points

    scaling_factor = 1
    x_shift = 0
    y_shift = 0
    rotation = 0

    bounding_size = 400
    offset = 50
    combined_matrix = np.zeros((bounding_size, bounding_size))

    base_directory = "/home/matthew/Documents/Elina_Data/"
    controls      = ["/Controls/Fish_1", "/Controls/Fish_2", "/Controls/Fish_3"]
    cortisol_high = ["/100uM/Fish_1", "/100uM/Fish_2", "/100uM/Fish_3"]
    cortisol_low  = ["40uM/Fish_1", "40uM/Fish_2", "40uM/Fish_3"]

    #Load Template Coords
    template_coords_file = base_directory + cortisol_low[2] + "/Tectum ROIs.csv"
    template_coords = load_roi_coordinates(template_coords_file)
    template_coords = np.add(template_coords, offset)


    fish_in_question = cortisol_high[2]

    #Load Coords To Match
    coords_to_match_file = base_directory + fish_in_question + "/Tectum ROIs.csv"
    coords_to_match = load_roi_coordinates(coords_to_match_file)
    coords_to_match = np.add(coords_to_match, offset)


    window_instance = tectum_matching_window()
    window_instance.display_axis.imshow(combined_matrix)
    window_instance.display_axis.scatter(template_coords[:, 0], template_coords[:, 1], c='b')

    window_instance.display_axis.scatter(coords_to_match[:, 0], coords_to_match[:, 1], c='g', alpha=0.3)

    sys.exit(app.exec_())

