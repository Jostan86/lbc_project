import numpy as np
import math
# import cv2
from dataset_info import get_dataset_info, image_to_gripper_transform, gripper_to_image_transform
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import layers
import numpy as np
import math
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import cv2

class ResultsProcessor:
    def __init__(self, model=None):

        self.dataset_info = get_dataset_info()
        self.gripper_bottom_row = self.dataset_info["gripper_bottom_row"]
        self.max_radius = self.dataset_info["max_radius"]
        self.gripper_right_column = self.dataset_info["gripper_right_column"]
        self.gripper_left_column = self.dataset_info["gripper_left_column"]
        self.pixels_per_mm = self.dataset_info["pixels_per_mm"]
        self.center_of_gripper_pixel = self.dataset_info["center_of_gripper_pixel"]
        self.gripper_center_row = self.dataset_info["gripper_center_row"]
        self.num_features = self.dataset_info["num_features"]
        self.num_time_steps = self.dataset_info["time_steps_to_use"]

        if model is None:
            self.model = tf.keras.models.load_model(self.dataset_info["model_path"])
        else:
            self.model = model
        # move distance in mm
        self.move_distance = self.dataset_info["move_distance"]

        self.move_right = True
        # Load the saved scaler
        scaler_path = self.dataset_info["scaler_path"]
        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        self.counter = 0

    @staticmethod
    def get_circle_center(left_point, right_point, radius):
        """ Finds the center of a circle given two points on the circle and the radius.

        Args:
            left_point (tuple): The starting point of the curve.
            right_point (tuple): The ending point of the curve.
            radius (float): The radius of the curve.

        Returns:
            circle_center (tuple): The center of the circle as (x, y) coordinates.
        """
        x_left, y_left = left_point
        x_right, y_right = right_point

        # Calculate the midpoint
        midpoint = ((x_left + x_right) / 2, (y_left + y_right) / 2)

        # Distance between start and end points
        dx, dy = x_right - x_left, y_right - y_left
        connecting_dist = math.sqrt(dx ** 2 + dy ** 2)
        points_to_center_dist = connecting_dist / 2
        dx_mid, dy_mid = dx / 2, dy / 2

        # Distance from midpoint to circle center
        if connecting_dist / 2 > abs(radius):
            print("The radius is too small for the given points, manually setting")
            # radius = connecting_dist / 2 * np.sign(radius)
            radius = 50 * np.sign(radius)

        mid_to_center_dist = math.sqrt(radius ** 2 - (connecting_dist / 2) ** 2)

        # if dx > 0:
        center_x1 = midpoint[0] + mid_to_center_dist * dy_mid / points_to_center_dist
        center_y1 = midpoint[1] - mid_to_center_dist * dx_mid / points_to_center_dist
        center_x2 = midpoint[0] - mid_to_center_dist * dy_mid / points_to_center_dist
        center_y2 = midpoint[1] + mid_to_center_dist * dx_mid / points_to_center_dist

        if radius < 0:
            center_x = center_x1
            center_y = center_y1
        else:
            center_x = center_x2
            center_y = center_y2

        return (center_x, center_y), radius

    @staticmethod
    def find_move_destination_points(center, radius, point, arc_length):
        """
        Finds two points on a circle that are a specified arc length away from a given point on the circle.

        Args:
            center (tuple): The center of the circle as (x, y) coordinates.
            radius (float): The radius of the circle.
            point (tuple): The point on the circle as (x, y) coordinates.
            arc_length (float): The arc length from the given point to the new points.

        Returns:
             Two tuples representing the coordinates of the new points on the circle in x, y coordinates.
        """
        radius_abs = abs(radius)
        # Unpack the center and point coordinates
        x_C, y_C = center
        x_P, y_P = point

        # Calculate the angle theta for the given arc length
        theta = arc_length / radius_abs

        # Calculate the initial angle phi for the given point
        phi = math.atan2(-(y_P - y_C), x_P - x_C)

        # Calculate new angles
        phi_1 = phi + theta
        phi_2 = phi - theta

        # Find the new points Q1 and Q2
        point_1 = (x_C + radius_abs * math.cos(phi_1), y_C - radius_abs * math.sin(phi_1))
        point_2 = (x_C + radius_abs * math.cos(phi_2), y_C - radius_abs * math.sin(phi_2))

        return point_1, point_2

    def get_result(self, x_data, transform=True):
        """ Gets the result of the model on a given input.

        Args:
            x_data (numpy.ndarray, dtype=float32, shape=(num_samples, num_time_steps, num_feature)): The input to the model.
            transform (bool): Whether or not to transform the input data using the saved scaler.

        Returns:
            prediction (numpy.ndarray, dtype=float32, shape=(num_samples, num_features)): The prediction of the model.
        """
        num_samples = x_data.shape[0]

        if transform:

            # reshape the data to 2D for scaling
            x_data = x_data.reshape(-1, self.num_features)
            x_data = self.scaler.transform(x_data)

        x_data = x_data.reshape(num_samples, self.num_time_steps, self.num_features)
        predictions = self.model.predict(x_data)

        return predictions


    def process_result(self, prediction, image, move_right=True, color=(0, 255, 0), ground_truth=False):

        prediction = prediction.copy()
        prediction[1:] /= self.dataset_info["enter_exit_multiplier"]

        curve_radius_pixels = (self.max_radius - abs(prediction[0]) * self.max_radius) * np.sign(prediction[0])
        left_exit_row = int(self.gripper_bottom_row - prediction[1] * self.gripper_bottom_row)
        right_exit_row = int(self.gripper_bottom_row - prediction[2] * self.gripper_bottom_row)

        left_exit_pixel_coords = (self.gripper_left_column, left_exit_row)
        right_exit_pixel_coords = (self.gripper_right_column, right_exit_row, )

        left_exit_xy = image_to_gripper_transform(left_exit_pixel_coords[1], left_exit_pixel_coords[0], self.dataset_info)
        right_exit_xy = image_to_gripper_transform(right_exit_pixel_coords[1], right_exit_pixel_coords[0], self.dataset_info)
        radius_mm = curve_radius_pixels / self.pixels_per_mm

        if abs(radius_mm) < self.dataset_info["min_radius"] and not ground_truth:
            radius_mm = self.dataset_info["min_radius"] * np.sign(radius_mm)
            self.counter += 1

        circle_center_xy, radius_mm = self.get_circle_center(left_exit_xy, right_exit_xy, radius_mm)

        move_xy, move_angle = self.get_move_command(circle_center_xy, left_exit_xy, right_exit_xy, radius_mm,
                                                    self.move_distance, move_right=move_right)

        image = self.draw_curve(image, circle_center_xy, left_exit_xy, right_exit_xy, radius_mm, color)

        # draw a dash at the right and left exit points on the image
        cv2.line(image, right_exit_pixel_coords, (right_exit_pixel_coords[0] + 15, right_exit_pixel_coords[1]), color, 2)
        cv2.line(image, left_exit_pixel_coords, (left_exit_pixel_coords[0] - 15, left_exit_pixel_coords[1]), color, 2)

        # draw a circle at the center of the gripper
        cv2.circle(image, (self.center_of_gripper_pixel, self.gripper_center_row), 5, color, -1)

        # Draw a circle at the destination point
        destination_point = gripper_to_image_transform(move_xy[0], move_xy[1], self.dataset_info)
        cv2.circle(image, (int(destination_point[1]), int(destination_point[0])), 5, color, -1)

        # Draw an arrow at the destination point
        arrow_angle = move_angle + 90
        arrow_start = (int(destination_point[1]), int(destination_point[0]))
        arrow_end_gripper_frame = (move_xy[0] + math.cos(math.radians(arrow_angle)) * 10,
                                      move_xy[1] + math.sin(math.radians(arrow_angle)) * 10)
        arrow_end = gripper_to_image_transform(arrow_end_gripper_frame[0], arrow_end_gripper_frame[1], self.dataset_info)
        arrow_end = (int(arrow_end[1]), int(arrow_end[0]))
        cv2.arrowedLine(image, arrow_start, arrow_end, color, 2)

        return move_xy, move_angle, image

    def get_move_command(self, circle_center, left_point, right_point, radius, arc_length, move_right):


        if move_right:
            point_1, point_2 = self.find_move_destination_points(circle_center, radius, right_point, arc_length)
            if point_1[0] > point_2[0]:
                destination_point = point_1
            else:
                destination_point = point_2
        else:
            point_1, point_2 = self.find_move_destination_points(circle_center, radius, left_point, arc_length)
            if point_1[0] < point_2[0]:
                destination_point = point_1
            else:
                destination_point = point_2

        dx = destination_point[0] - circle_center[0]
        dy = destination_point[1] - circle_center[1]
        # destination angle is the angle between the destination point and the circle center
        if radius > 0:
            destination_angle = math.degrees(math.atan2(dx, -dy))
        else:
            destination_angle = math.degrees(math.atan2(-dx, dy))

        return destination_point, destination_angle

    def draw_curve(self, image, circle_center_xy, left_exit_xy, right_exit_xy, radius, color):
        """ Draws a curve on an image given the center and radius of the circle.

        Args:
            image (numpy.ndarray, dtype=uint8, shape=(480, 640, 3)): The image to draw the curve on.
            circle_center_xy (tuple): The center of the circle as (x, y) coordinates in the gripper frame.
            left_exit_xy (tuple): The left exit point of the curve as (x, y) coordinates in the gripper frame.
            right_exit_xy (tuple): The ending point of the curve as (x, y) coordinates in the gripper frame.
            radius (float): The radius of the curve in mm.
            color (tuple): The color to make the curve.

        Returns:
            image (numpy.ndarray, dtype=uint8, shape=(480, 640, 3)): The image with the drawn curve.
        """

        # Determine the start and end angles for the arc
        left_angle = math.degrees(math.atan2(left_exit_xy[1] - circle_center_xy[1],
                                             left_exit_xy[0] - circle_center_xy[0]))
        right_angle = math.degrees(math.atan2(right_exit_xy[1] - circle_center_xy[1],
                                              right_exit_xy[0] - circle_center_xy[0]))


        # Adjust angle to be between 0 and 360
        if left_angle < 0:
            left_angle += 360
        if right_angle < 0:
            right_angle += 360

        right_angle = 360 - right_angle
        left_angle = 360 - left_angle

        # Adjust angle in the case that the curve needs to go over 0 degrees
        if radius < 0 and right_exit_xy[1] < circle_center_xy[1]:
            right_angle += 360
        if radius > 0 and right_exit_xy[1] > circle_center_xy[1]:
            right_angle += 360

        # Convert the circle center to pixel coordinates
        circle_center_pix = gripper_to_image_transform(circle_center_xy[0], circle_center_xy[1], self.dataset_info)

        radius_pix = radius * self.pixels_per_mm

        # Draw the arc
        cv2.ellipse(image,
                    (int(circle_center_pix[1]), int(circle_center_pix[0])),
                    (int(abs(radius_pix)), int(abs(radius_pix))),
                    0,
                    right_angle,
                    left_angle,
                    color,
                    2)

        return image





if __name__ == "__main__":
    ...