import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from process_data_jostan import get_dataset_info
import cv2
import math
import os

def train_model(epochs, lr):
    """ Trains the LSTM model and saves it. Also plots the training and validation loss."""

    dataset_info = get_dataset_info()

    X_train = np.load(dataset_info["X_train_path"])
    y_train = np.load(dataset_info["y_train_path"])

    # Define the LSTM model with multiple layers
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(50, activation='relu', input_shape=(dataset_info["time_steps_to_use"], dataset_info["num_features"])))

    model.add(Dense(y_train.shape[1]))  # Output layer: Predicts the radius

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # Adjust the learning rate

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Assuming X_train and y_train are your input and output training data
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=100)

    # Plot loss from training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # # Plot accuracy from training history
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    # save the model
    model.save(dataset_info["model_path"])

    return model

def get_circle_center(start_point, end_point, radius):
    """ Finds the center of a circle given two points on the circle and the radius.

    Args:
        start_point (tuple): The starting point of the curve.
        end_point (tuple): The ending point of the curve.
        radius (float): The radius of the curve.

    Returns:
        circle_center (tuple): The center of the circle as (x, y) pixel coordinates.
    """

    # Calculate the midpoint
    midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)

    # Distance between start and end points
    dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]
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

def draw_curve(image, circle_center, start_point, end_point, radius, color):
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
    start_angle = math.degrees(math.atan2(start_point[1] - circle_center[1], start_point[0] - circle_center[0]))
    end_angle = math.degrees(math.atan2(end_point[1] - circle_center[1], end_point[0] - circle_center[0]))

    # if distance between angles is over 180, switch the angles
    switch_angles = False
    if abs(start_angle - end_angle) > 175:
        switch_angles = True

    if switch_angles:
        # add 360 to start or end angle, whichever is negative
        if start_angle < 0:
            start_angle += 360
        else:
            end_angle += 360

    # Draw the arc
    cv2.ellipse(image,
                (int(circle_center[0]), int(circle_center[1])),
                (int(abs(radius)), int(abs(radius))),
                0,
                end_angle,
                start_angle,
                color,
                2)

    return image

def draw_poly(image, start_point, end_point, poly_coeffs, color):
    """Plots a polynomial curve over an input image.

    Args:
        image (numpy.ndarray): The image to draw the polynomial curve on.
        start_point (tuple): The starting point of the curve as (x, y) pixel coordinates.
        end_point (tuple): The ending point of the curve as (x, y) pixel coordinates.
        poly_coeffs (numpy.ndarray): Coefficients of the polynomial curve.
        color (tuple): The color of the curve.

    Returns:
        image (numpy.ndarray): The image with the drawn polynomial curve.
    """

    x_vals = np.linspace(start_point[0], end_point[0], 100)
    y_vals = np.polyval(poly_coeffs, x_vals)

    # Scale y_vals based on start_point[1] and end_point[1]
    y_range = end_point[1] - start_point[1]
    y_vals_scaled = ((y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals))) * y_range + start_point[1]

    # Check the direction of the curve
    if start_point[1] > end_point[1]:
        # If the start_point is below the end_point, flip the curve
        y_vals_scaled = end_point[1] - (y_vals_scaled - end_point[1])

    # Create curve points using scaled y_vals
    curve_points = np.column_stack((x_vals.astype(np.int32), y_vals_scaled.astype(np.int32)))

    # Draw the polynomial curve on the image
    cv2.polylines(image, [curve_points], isClosed=False, color=color, thickness=2)

    return image

def annotate_image_with_result(image, result, color=(0, 0, 255), segmentation_method='radius'):

    dataset_info = get_dataset_info()

    bottom_pixel = dataset_info["gripper_bottom_row"]
    max_radius = dataset_info["max_radius"]
    min_orig_coeff = np.array(dataset_info["min_orig_coeff"])
    max_orig_coeff = np.array(dataset_info["max_orig_coeff"])
    right_column = dataset_info["gripper_right_column"]
    left_column = dataset_info["gripper_left_column"]

    if segmentation_method == 'radius':
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

    if segmentation_method == 'poly':
        # convert exits back to pixels
        result[3:] = bottom_pixel - result[3:] * bottom_pixel
        left_exit_row = int(result[3])
        right_exit_row = int(result[4])
        pixel_coeff = np.array(result[0:3]) * max_orig_coeff 
        left_exit_coords = (left_column, left_exit_row)
        right_exit_coords = (right_column, right_exit_row)

        # Draw a line along the curve
        image = draw_poly(image, left_exit_coords, right_exit_coords, pixel_coeff, color)

    # draw a dash at the right and left exit points on the image
    cv2.line(image, (right_column, right_exit_row), (right_column + 15, right_exit_row), color, 2)
    cv2.line(image, (left_column, left_exit_row), (left_column - 15, left_exit_row), color, 2)


def test_model(model=None, overwrite=False, segmentation_method='radius'):
    """ Tests the model on the test data and saves the results as annotated images.

    Args:
        model (tensorflow.keras.models.Sequential): The model to test. If None, the model will be loaded from the dataset directory.
        overwrite (bool): If True, the results directory will be overwritten. If False, an error will be raised if the results directory is not empty.
    """

    dataset_info = get_dataset_info()

    results_save_path = os.path.join(dataset_info["dataset_directory"], "results1")

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

        annotate_image_with_result(image, prediction, (57, 127, 255), segmentation_method)
        annotate_image_with_result(image, y_test_sample, (255, 204, 52), segmentation_method)

        # save the image
        cv2.imwrite(os.path.join(results_save_path, "result_{}.png".format(image_name.split(".")[0])), image)

        # Set numpy to print 2 decimal places
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        print("Prediction:", prediction, "Actual:", y_test_sample)


if __name__ == "__main__":
    model = train_model(epochs=750, lr=0.0001)
    test_model(overwrite=True, segmentation_method='poly')

