#!/usr/bin/env python3

import rospy
from dataset_info import get_dataset_info
from lbc_project.srv import MoveData, MoveDataResponse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import layers
import numpy as np
import math
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

class MoveServer:
    def __init__(self):
        rospy.init_node('move_server')
        self.dataset_info = get_dataset_info()
        self.model = tf.keras.models.load_model(self.dataset_info["model_path"])

        self.gripper_bottom_row = self.dataset_info["gripper_bottom_row"]
        self.max_radius = self.dataset_info["max_radius"]
        self.gripper_right_column = self.dataset_info["gripper_right_column"]
        self.gripper_left_column = self.dataset_info["gripper_left_column"]

        self.num_features = self.dataset_info["num_features"]
        self.num_time_steps = self.dataset_info["time_steps_to_use"]

        # move distance in mm
        move_distance = self.dataset_info["move_distance"]
        self.move_distance = move_distance * self.dataset_info["pixels_per_mm"]

        self.move_right = True
        # Load the saved scaler
        with open('/home/jostan/catkin_ws/src/pkgs_noetic/course_pkgs/lbc/lbc_project/src/scaler.pkl', 'rb') as file:
            self.scaler = pickle.load(file)

        self.move_data_service = rospy.Service('gripper_move_data', MoveData, self.return_move_data)

        # Normalize new data
        # new_og_shape = (1, self.num_time_steps, self.num_features)
        # new_reshaped_data = X_new.reshape(-1, X_new.shape[-1])
        # normalized_new_data = self.scaler.transform(new_reshaped_data)
        # X_new_normalized = normalized_new_data.reshape(new_og_shape)


        rospy.spin()

    def load_data_service(self, request):


        path_to_data = "/home/jostan/Documents/lbc_datasets/lbc_dataset4/grip_data/0_0.csv"

        gripper_data = pd.read_csv(path_to_data)

        # Extract the last 30 rows and drop non-feature columns if necessary
        x_data = gripper_data.iloc[-self.dataset_info['time_steps_to_use']:].drop(columns=self.dataset_info['non_feature_columns']).values

        x_data = self.scaler.transform(x_data)

        return self.return_move_data(x_data)

    def receive_data_service(self, request):

        x_data = request.gripper_data
        self.move_right = request.move_right
        x_data = np.array(x_data).reshape(1, self.num_time_steps, self.num_features)[0]
        return self.return_move_data(x_data)

    def return_move_data(self, request):

        # path_to_data = "/home/jostan/Documents/lbc_datasets/lbc_dataset4/grip_data/0_0.csv"
        #
        # gripper_data = pd.read_csv(path_to_data)
        #
        # # Extract the last 30 rows and drop non-feature columns if necessary
        # x_data = gripper_data.iloc[-self.dataset_info['time_steps_to_use']:].drop(
        #     columns=self.dataset_info['non_feature_columns']).values

        x_data = request.gripper_data

        x_data = np.array(x_data).reshape(self.num_time_steps, self.num_features)

        x_data = self.scaler.transform(x_data).reshape(1, self.num_time_steps, self.num_features)

        prediction = self.model.predict(x_data)[0]
        prediction[1:] = self.gripper_bottom_row - prediction[1:] * self.gripper_bottom_row
        left_exit_row = int(prediction[1])
        right_exit_row = int(prediction[2])
        curve_radius_pixels = (self.max_radius - abs(prediction[0]) * self.max_radius) * np.sign(prediction[0])
        left_exit_coords = (self.gripper_left_column, left_exit_row)
        right_exit_coords = (self.gripper_right_column, right_exit_row)

        move_data = self.get_move(left_exit_coords, right_exit_coords, curve_radius_pixels, self.move_distance, self.move_right)

        response = MoveDataResponse()
        response.x = move_data[0]
        response.y = move_data[1]
        response.angle = move_data[2]
        return response

    def get_move(self, left_point, right_point, radius, arc_length, move_right):

        destination_point, destination_angle = self.get_move_destination(left_point, right_point, radius, arc_length,
                                                                    move_right=move_right)

        gripper_center_point = (self.dataset_info["center_of_gripper_pixel"], self.dataset_info["gripper_center_row"])

        dx_pix = destination_point[0] - gripper_center_point[0]
        dy_pix = -(destination_point[1] - gripper_center_point[1])
        # d theta is the angle between

        pix_per_mm = self.dataset_info["pixels_per_mm"]
        dx_mm = dx_pix / pix_per_mm
        dy_mm = dy_pix / pix_per_mm

        return (dx_mm, dy_mm, destination_angle)
    def get_move_destination(self, left_point, right_point, radius, arc_length, move_right):
        circle_center = self.get_circle_center(left_point, right_point, radius)[0]

        if move_right:
            Q1, Q2 = self.find_points_on_circle(circle_center, radius, right_point, arc_length)
            if Q1[0] > Q2[0]:
                destination_point = Q1
            else:
                destination_point = Q2
        else:
            Q1, Q2 = self.find_points_on_circle(circle_center, radius, left_point, arc_length)
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

    def find_points_on_circle(self, center, radius, point, arc_length):
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

        # Calculate the initial angle phi for the given point
        phi = math.atan2(-(y_P - y_C), x_P - x_C)

        # Calculate new angles
        phi_1 = phi + theta
        phi_2 = phi - theta

        # Find the new points Q1 and Q2
        Q1 = (x_C + radius_abs * math.cos(phi_1), y_C - radius_abs * math.sin(phi_1))
        Q2 = (x_C + radius_abs * math.cos(phi_2), y_C - radius_abs * math.sin(phi_2))

        return Q1, Q2

    def get_circle_center(self, left_point, right_point, radius):
        """ Finds the center of a circle given two points on the circle and the radius.

        Args:
            start_point (tuple): The starting point of the curve.
            end_point (tuple): The ending point of the curve.
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

if __name__ == "__main__":
    MoveServer()


