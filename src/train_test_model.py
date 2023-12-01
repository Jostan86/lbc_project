import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
from process_data import get_dataset_info
import cv2
import math
import os

def train_model():
    """ Trains the LSTM model and saves it. Also plots the training and validation loss."""

    dataset_info = get_dataset_info()

    X_train = np.load(dataset_info["X_train_path"])
    y_train = np.load(dataset_info["y_train_path"])

    # Define the LSTM model with multiple layers
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(50, activation='relu', input_shape=(dataset_info["time_steps_to_use"], dataset_info["num_features"])))

    model.add(Dense(3))  # Output layer: Predicts the radius

    model.compile(optimizer='adam', loss='mean_squared_error')

    # shuffle the data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]


    # Assuming X_train and y_train are your input and output training data
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.1)

    # Plot accuracy from training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # save the model
    model.save(dataset_info["model_path"])

    return model

def get_circle_center(left_point, right_point, radius):
    """ Finds the center of a circle given two points on the circle and the radius.

    Args:
        left_point (tuple): The starting point of the curve.
        right_point (tuple): The ending point of the curve.
        radius (float): The radius of the curve.

    Returns:
        circle_center (tuple): The center of the circle as (x, y) pixel coordinates.
    """

    # Calculate the midpoint
    midpoint = ((left_point[0] + right_point[0]) / 2, (left_point[1] + right_point[1]) / 2)

    # Distance between start and end points
    dx, dy = right_point[0] - left_point[0], right_point[1] - left_point[1]
    connecting_dist = math.sqrt(dx ** 2 + dy ** 2)
    points_to_center_dist = connecting_dist / 2
    dx_mid, dy_mid = dx / 2, dy / 2

    # Distance from midpoint to circle center
    if connecting_dist / 2 > abs(radius):
        print("The radius is too small for the given points, manually setting")
        radius = connecting_dist / 2 * np.sign(radius)
    mid_to_center_dist = math.sqrt(radius ** 2 - (connecting_dist / 2) ** 2)

    # if dx > 0:
    center_x1 = midpoint[0] + mid_to_center_dist * dy_mid / points_to_center_dist
    center_y1 = midpoint[1] - mid_to_center_dist * dx_mid / points_to_center_dist
    center_x2 = midpoint[0] - mid_to_center_dist * dy_mid / points_to_center_dist
    center_y2 = midpoint[1] + mid_to_center_dist * dx_mid / points_to_center_dist

    if radius > 0:
        center_x = center_x1
        center_y = center_y1
    else:
        center_x = center_x2
        center_y = center_y2

    return (center_x, center_y), radius

def draw_curve(image, circle_center, left_point, right_point, radius, color):
    """ Draws a curve on an image given the center and radius of the circle.

    Args:
        image (numpy.ndarray, dtype=uint8, shape=(480, 640, 3)): The image to draw the curve on.
        circle_center (tuple): The center of the circle as (x, y) pixel coordinates.
        start_point (tuple): The starting point of the curve as (x, y) pixel coordinates.
        end_point (tuple): The ending point of the curve as (x, y) pixel coordinates.
        radius (float): The radius of the curve.
        color (tuple): The color of the curve.

    Returns:
        image (numpy.ndarray, dtype=uint8, shape=(480, 640, 3)): The image with the drawn curve.
    """

    # Determine the start and end angles for the arc
    left_angle = math.degrees(math.atan2(left_point[1] - circle_center[1], left_point[0] - circle_center[0]))
    right_angle = math.degrees(math.atan2(right_point[1] - circle_center[1], right_point[0] - circle_center[0]))

    # if distance between angles is over 180, switch the angles
    switch_angles = False
    if abs(left_angle - right_angle) > 175:
        switch_angles = True

    if switch_angles:
        # add 360 to start or end angle, whichever is negative
        if left_angle < 0:
            left_angle += 360
        else:
            right_angle += 360


    # Draw the arc
    cv2.ellipse(image,
                (int(circle_center[0]), int(circle_center[1])),
                (int(abs(radius)), int(abs(radius))),
                0,
                right_angle,
                left_angle,
                color,
                2)

    return image

def annotate_image_with_result(image, result, color=(0, 0, 255)):

    dataset_info = get_dataset_info()

    bottom_pixel = dataset_info["gripper_bottom_row"]
    max_radius = dataset_info["max_radius"]
    right_column = dataset_info["gripper_right_column"]
    left_column = dataset_info["gripper_left_column"]

    # convert exits back to pixels
    result[1:] = bottom_pixel - result[1:] * bottom_pixel
    left_exit_row = int(result[1])
    right_exit_row = int(result[2])
    curve_radius_pixel = (max_radius - abs(result[0]) * max_radius) * np.sign(result[0])
    left_exit_coords = (left_column, left_exit_row)
    right_exit_coords = (right_column, right_exit_row)

    # Draw a line along the curve
    circle_center, curve_radius_pixel = get_circle_center(left_exit_coords, right_exit_coords, curve_radius_pixel)
    image = draw_curve(image, circle_center, left_exit_coords, right_exit_coords, curve_radius_pixel, color)

    # draw a dash at the right and left exit points on the image
    cv2.line(image, (right_column, right_exit_row), (right_column + 15, right_exit_row), color, 2)
    cv2.line(image, (left_column, left_exit_row), (left_column - 15, left_exit_row), color, 2)

    arc_length = 80

    destination, destination_angle = get_move_destination(left_exit_coords, right_exit_coords, curve_radius_pixel,
                                                     arc_length,
                                        move_right=True)
    cv2.circle(image, (int(destination[0]), int(destination[1])), 5, color, -1)

    print(get_move_data(left_exit_coords, right_exit_coords, curve_radius_pixel, arc_length, dataset_info,
                        move_right=True))

import math

def test_model(model=None, overwrite=False):
    """ Tests the model on the test data and saves the results as annotated images.

    Args:
        model (tensorflow.keras.models.Sequential): The model to test. If None, the model will be loaded from the dataset directory.
        overwrite (bool): If True, the results directory will be overwritten. If False, an error will be raised if the results directory is not empty.
    """

    dataset_info = get_dataset_info()

    results_save_path = os.path.join(dataset_info["dataset_directory"], "results2")

    if not os.path.exists(results_save_path):
        raise ValueError("The results directory does not exist.")

    if not overwrite and os.listdir(results_save_path) != []:
        raise ValueError("The results directory is not empty, set overwrite to True to overwrite the directory.")
    else:
        for file in os.listdir(results_save_path):
            os.remove(os.path.join(results_save_path, file))

    if model is None:
        model = tf.keras.models.load_model(dataset_info["model_path"])

    X_test = np.load(dataset_info["X_test_path"])
    y_test = np.load(dataset_info["y_test_path"])
    file_names_test = np.load(dataset_info["test_data_file_names_path"])

    predictions = model.predict(X_test)

    for i, (prediction, y_test_sample, image_name) in enumerate(zip(predictions, y_test, file_names_test)):

        print("Image:", image_name)

        image_path = os.path.join(dataset_info["image_directory"], image_name + ".jpg")

        image = cv2.imread(image_path)

        annotate_image_with_result(image, prediction, (57, 127, 255))
        annotate_image_with_result(image, y_test_sample, (255, 204, 52))



        # save the image
        cv2.imwrite(os.path.join(results_save_path, "result_{}.png".format(image_name.split(".")[0])), image)

        # Set numpy to print 2 decimal places
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        print("Prediction:", prediction, "Actual:", y_test_sample)

def find_points_on_circle(center, radius, point, arc_length):
    """
    Finds two points on a circle that are a specified arc length away from a given point on the circle.

    Args:
        center (tuple): The center of the circle as (x, y) pixel coordinates.
        radius (float): The radius of the circle.
        point (tuple): The point on the circle as (x, y) pixel coordinates.
        arc_length (float): The arc length from the given point to the new points.

    Returns:
         Two tuples representing the coordinates of the new points on the circle.
    """
    radius_abs = abs(radius)
    # Unpack the center and point coordinates
    x_C, y_C = center
    x_P, y_P = point

    # Calculate the angle theta for the given arc length
    theta = arc_length / radius_abs

    # Calculate the initial angle phi for the given point, adjusting for the image coordinate system
    phi = math.atan2(-(y_P - y_C), x_P - x_C) # y difference is negated

    # Calculate new angles
    phi_1 = phi + theta
    phi_2 = phi - theta

    # Find the new points Q1 and Q2
    Q1 = (x_C + radius_abs * math.cos(phi_1), y_C - radius_abs * math.sin(phi_1)) # y is subtracted
    Q2 = (x_C + radius_abs * math.cos(phi_2), y_C - radius_abs * math.sin(phi_2)) # y is subtracted


    return Q1, Q2

def get_move_destination(left_point, right_point, radius, arc_length, move_right):
    circle_center = get_circle_center(left_point, right_point, radius)[0]

    if move_right:
        Q1, Q2 = find_points_on_circle(circle_center, radius, right_point, arc_length)
        if Q1[0] > Q2[0]:
            destination_point = Q1
        else:
            destination_point = Q2
    else:
        Q1, Q2 = find_points_on_circle(circle_center, radius, left_point, arc_length)
        if Q1[0] < Q2[0]:
            destination_point = Q1
        else:
            destination_point = Q2

    dx = destination_point[0] - circle_center[0]
    dy = -(destination_point[1] - circle_center[1])
    # destination angle is the angle between the destination point and the circle center
    if radius > 0:
        destination_angle = math.degrees(math.atan2(dx, -dy))
    else:
        destination_angle = math.degrees(math.atan2(-dx, dy))
    return destination_point, destination_angle

def get_move_data(left_point, right_point, radius, arc_length, dataset_info, move_right=True):

    destination_point, destination_angle = get_move_destination(left_point, right_point, radius, arc_length,
                                                                move_right=move_right)

    gripper_center_point = (dataset_info["center_of_gripper_pixel"], dataset_info["gripper_center_row"])

    dx_pix = destination_point[0] - gripper_center_point[0]
    dy_pix = -(destination_point[1] - gripper_center_point[1])
    # d theta is the angle between

    pix_per_mm = dataset_info["pixels_per_mm"]
    dx_mm = dx_pix / pix_per_mm
    dy_mm = dy_pix / pix_per_mm

    return (dx_mm, dy_mm, destination_angle)



if __name__ == "__main__":
    model = train_model()
    test_model(overwrite=True)



