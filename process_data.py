import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import copy
from skimage.morphology import medial_axis, remove_small_objects
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataset_info import get_dataset_info

class GroundTruther:
    """ Class for generating ground truth data for the radius and enter/exit points of the wire."""

    def __init__(self, viz=False, overwrite=False):
        """ Initialize
        Args:
            viz (bool): True if you want to visualize the skeleton and enter/exit point images
            overwrite (bool): True if you want to overwrite the images in the save directories
        """

        data_set_info = get_dataset_info()

        self.captioned_image_save_directory = data_set_info["annotated_image_directory"]

        self.max_radius = data_set_info["max_radius"]
        self.max_coeff = data_set_info["max_orig_coeff"]

        self.x_range = (data_set_info["gripper_left_column"], data_set_info["gripper_right_column"])

        self.center_of_gripper_pixel = data_set_info["center_of_gripper_pixel"]
        self.center_segment_half_width = data_set_info["center_segment_half_width"]

        self.bottom_crop = data_set_info["bottom_crop"]
        self.top_crop = data_set_info["top_crop"]

        self.gripper_right_column = data_set_info["gripper_right_column"]
        self.gripper_left_column = data_set_info["gripper_left_column"]
        self.gripper_bottom_row = data_set_info["gripper_bottom_row"]

        self.viz = viz
        self.overwrite = overwrite

        self.empty_save_directories()

    def empty_save_directories(self):
        """ Empties the save directories if they exist and overwrite is True. Raises an error if they don't exist."""
        if self.captioned_image_save_directory is not None:
            if not os.path.exists(self.captioned_image_save_directory):
                raise FileNotFoundError("The directory {} does not exist.".format(
                    self.captioned_image_save_directory))
            if len(os.listdir(self.captioned_image_save_directory)) != 0 and not self.overwrite:
                raise ValueError("The directory {} is not empty. Set overwrite to True to overwrite".format(
                    self.captioned_image_save_directory))
            for file in os.listdir(self.captioned_image_save_directory):
                os.remove(os.path.join(self.captioned_image_save_directory, file))

    def fit_circle(self, x_coords, y_coords):
        """ Fits a circle to the given points and returns the center and radius of the circle.

        Args:
            x_coords (np.array, dtype=np.int): x coordinates of the points
            y_coords (np.array, dtype=np.int): y coordinates of the points

        Returns:
            circle_center_x (float): x coordinate of the circle center
            circle_center_y (float): y coordinate of the circle center
            radius (float): radius of the circle
            """
        # Combining x and y into a single array
        points = np.vstack((x_coords, y_coords)).T

        # Formulating the matrix equation
        A = np.hstack((points, np.ones((points.shape[0], 1))))
        b = np.sum(points**2, axis=1).reshape(-1, 1)

        # Solving the least squares problem
        c = np.linalg.lstsq(A, b, rcond=None)[0]

        # Extracting circle parameters
        circle_center_x, circle_center_y = c[0:2].flatten() / 2
        radius = np.sqrt(c[2] + circle_center_x**2 + circle_center_y**2)
        return circle_center_x, circle_center_y, radius

    def prune_skeleton(self, skeleton, branch_min_size=20):
        """ Removes small branches from the skeleton. May also remove chunks of the skeleton if there are a lot of
        branches... so use cautiously.

        Args:
            skeleton (np.array, dtype=np.bool, shape=(480, 640)): skeleton of the image as a 2D boolean array
            branch_min_size (int): minimum size of a branch to keep in pixels

        Returns:
            pruned_skeleton (np.array, dtype=np.bool, shape=(480, 640)): pruned skeleton of the image as a 2D boolean array
        """

        # Identify branch points
        branch_points = np.zeros_like(skeleton)
        branch_point_coords = []
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j] == 1:
                    # Get the 3x3 neighborhood
                    neighborhood = skeleton[i - 1:i + 2, j - 1:j + 2]
                    # Sum the neighborhood
                    neighborhood_sum = np.sum(neighborhood)
                    # If the sum is greater than 3, then it is a branch point
                    if neighborhood_sum > 3:
                        branch_points[i, j] = 1
                        branch_point_coords.append((i, j))

        # Remove branch points
        pruned_skeleton = np.logical_and(skeleton, ~branch_points)

        # Remove any small 'chunks' of the skeleton
        pruned_skeleton = remove_small_objects(pruned_skeleton, min_size=branch_min_size, connectivity=2)

        # Add back in the branch points
        pruned_skeleton = np.logical_or(pruned_skeleton, branch_points)

        return pruned_skeleton

    def get_red_mask(self, image_bgr):
        """ Returns a mask of the red pixels in the image

        Args:
            image_bgr (np.array, dtype=np.uint8, shape=(480, 640, 3)): image as a BGR numpy array

        Returns:
            red_mask (np.array, dtype=np.uint8, shape=(480, 640)): mask of the red pixels in the image
        """

        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 100, 25])
        upper_red_1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(image_hsv, lower_red_1, upper_red_1)

        # Additional range for red color
        lower_red_2 = np.array([150, 100, 25])
        upper_red_2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(image_hsv, lower_red_2, upper_red_2)

        # Combine masks
        red_mask = mask1 + mask2

        return red_mask

    def get_skeleton(self, image_bgr, image_trial_num=None):
        """ Finds the skeleton of the wire.

        Args:
            image_bgr (np.array, dtype=np.uint8, shape=(480, 640, 3)): image as a BGR numpy array
            image_trial_num (int): trial number of the image for saving purposes

        Returns:
            cropped_pruned_skeleton (np.array, dtype=np.bool, shape=(480, 640)): skeleton of the image as a 2D boolean array
        """

        red_mask = self.get_red_mask(image_bgr)

        kernel = np.ones((5, 5), np.uint8)
        morphed_red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        masked_image_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=morphed_red_mask)

        gray_masked_image = cv2.cvtColor(masked_image_bgr, cv2.COLOR_BGR2GRAY)

        blurred_gray_masked_image = cv2.GaussianBlur(gray_masked_image, (25, 25), sigmaX=22)

        mask_of_blurred_image = cv2.inRange(blurred_gray_masked_image, 10, 255)

        skeleton, return_distance = medial_axis(mask_of_blurred_image, return_distance=True)

        pruned_skeleton = self.prune_skeleton(skeleton)

        cropped_pruned_skeleton = copy.copy(pruned_skeleton)

        cropped_pruned_skeleton[-self.bottom_crop:, :] = 0
        cropped_pruned_skeleton[:self.top_crop, :] = 0
        cropped_pruned_skeleton[:, :self.center_of_gripper_pixel - self.center_segment_half_width] = 0
        cropped_pruned_skeleton[:, self.center_of_gripper_pixel + self.center_segment_half_width:] = 0

        image_bgr[cropped_pruned_skeleton == 1] = [255, 0, 0]

        if self.viz:
            # display skeleton
            if image_trial_num is not None:
                print("image_trial_num: {}".format(str(image_trial_num)))
            cv2.imshow('image', image_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if image_trial_num is None and self.captioned_image_save_directory is not None:
            raise ValueError("image_trial_num must be specified for skeleton image to be saved")
        elif self.captioned_image_save_directory is not None:
            cv2.imwrite(os.path.join(self.captioned_image_save_directory, "{}_skeleton.png".format(image_trial_num)),
                        image_bgr)

        return cropped_pruned_skeleton

    def get_radius(self, image_bgr, skeleton, image_trial_num=None):
        """ Finds the radius of curvature of the wire.

        Args:
            image_bgr (np.array, dtype=np.uint8, shape=(480, 640, 3)): image as a BGR numpy array
            skeleton (np.array, dtype=np.bool, shape=(480, 640)): skeleton of the cable as a 2D boolean array
            image_trial_num (int): trial number of the image for saving purposes

        Returns:
            circle_radius (float): radius of curvature of the wire
        """

        skeleton_coordinates = np.where(skeleton == 1)

        skeleton_x_coords = skeleton_coordinates[1]
        skeleton_y_coords = skeleton_coordinates[0]

        circle_center_x, circle_center_y, circle_radius = self.fit_circle(skeleton_x_coords, skeleton_y_coords)

        # Draw a line at the center of the gripper and a circle showing the radius of curvature
        captioned_image = image_bgr.copy()
        cv2.line(captioned_image,
                 (self.center_of_gripper_pixel, 0),
                 (self.center_of_gripper_pixel, captioned_image.shape[0]),
                 (0, 0, 255),
                 1)
        cv2.circle(captioned_image,
                   (int(circle_center_x), int(circle_center_y)),
                   int(circle_radius),
                   (255, 0, 0),
                   1)

        if self.viz:
            if image_trial_num is not None:
                print("image_trial_num: {}".format(str(image_trial_num)))
            cv2.imshow('image', captioned_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if image_trial_num is None and self.captioned_image_save_directory is not None:
            raise ValueError("image_trial_num must be specified for skeleton image to be saved")
        elif self.captioned_image_save_directory is not None:
            cv2.imwrite(os.path.join(self.captioned_image_save_directory, "{}_radius.png".format(image_trial_num)),
                        captioned_image)

        try:
            skeleton_center_row_index = np.where(skeleton[:, self.center_of_gripper_pixel] == 1)[0][0]
        except IndexError as e:
            print("IndexError: {}".format(e))
            if image_trial_num is not None:
                print("image_trial_num: {}".format(str(image_trial_num)))
            print("Skeleton may be lacking a point in the column along the center of the gripper")
            raise


        # Reduce the radius if it's close to the max radius (-1 is to avoid some bugs)
        circle_radius = self.max_radius - 1 if circle_radius >= self.max_radius else circle_radius[0]

        # Adjust the radius to that it's between 1 and -1, where one is a tight curve up, -1 is a tight curve down,
        # and 0 is a flatish line
        if circle_center_y > skeleton_center_row_index:
            circle_radius = -self.max_radius + circle_radius
        else:
            circle_radius = self.max_radius - circle_radius

        # interpolate the save radius to be between 1 and -1
        circle_radius = (circle_radius + self.max_radius) / (2 * self.max_radius) * 2 - 1

        return circle_radius

    def get_poly(self, image_bgr, skeleton, image_trial=None, degree=2):
            """Fits a polynomial to the curvature of the wire and outputs the polynomial's coefficients.

            Args:
                image_bgr (np.array, dtype=np.uint8, shape=(480, 640, 3)): image as a BGR numpy array
                skeleton (np.array, dtype=np.bool, shape=(480, 640)): skeleton of the cable as a 2D boolean array
                image_trial (int): trial number of the image for saving purposes
                degree (int): degree of the polynomial to fit (default: 3)

            Returns:
                coefficients (np.array): coefficients of the fitted polynomial
            """

            skeleton_coordinates = np.where(skeleton == 1)
            skeleton_x_coords = skeleton_coordinates[1]
            skeleton_y_coords = skeleton_coordinates[0]

            # Filter skeleton points within the specified x-range
            x_mask = np.where((skeleton_x_coords >= self.x_range[0]) & (skeleton_x_coords <= self.x_range[1]))
            filtered_x_coords = skeleton_x_coords[x_mask]
            filtered_y_coords = skeleton_y_coords[x_mask]

            # Fit a polynomial curve to the filtered skeleton points
            coefficients = np.polyfit(filtered_x_coords, filtered_y_coords, degree)

            # Normalize the coefficients to be between -1 and 1
            normalized_coefficients = coefficients / self.max_coeff

            # Visualize the fitted polynomial within the specified x-range (optional)
            if self.viz:
                fitted_curve = np.poly1d(coefficients)
                x_values = np.linspace(self.x_range[0], self.x_range[1], 100)  # Generate x-values within the range
                y_values = fitted_curve(x_values)  # Calculate corresponding y-values
                
                curve_points = np.column_stack((x_values.astype(np.int32), y_values.astype(np.int32)))

                # Draw the polynomial curve on the image
                cv2.polylines(image_bgr, [curve_points], isClosed=False, color=(255, 0, 0), thickness=2)
                cv2.imshow("image", image_bgr)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return normalized_coefficients

    def get_enter_exit_point(self, image_bgr, image_save_prefix=None):
        """ Finds the enter and exit points of the wire. They are normalized to be between 0 and 1, where 0 is the
        bottom of the gripper and 1 is the top of the image. The bottom row of the gripper is set in the __init__ method.

        Args:
            image_bgr (np.array, dtype=np.uint8, shape=(480, 640, 3)): image as a BGR numpy array
            image_save_prefix (str): prefix for the image name if you want to save the image

        Returns:
            right_exit (float): normalized exit point of the right side of the wire
            left_exit (float): normalized exit point of the left side of the wire
        """

        red_mask = self.get_red_mask(image_bgr)

        right_side_masked_rows = []
        left_side_masked_rows = []

        gripper_left_column = self.gripper_left_column
        gripper_right_column = self.gripper_right_column


        try:
            while len(right_side_masked_rows) < 1:
                right_column_rows = np.where(red_mask[:, gripper_right_column] > 0)[0]
                right_side_masked_rows += list(right_column_rows)
                gripper_right_column += 1

            while len(left_side_masked_rows) < 1:
                left_column_rows = np.where(red_mask[:, gripper_left_column] > 0)[0]
                left_side_masked_rows += list(left_column_rows)
                gripper_left_column -= 1
        except IndexError as e:
            print("IndexError: {}".format(e))
            if image_save_prefix is not None:
                print("image: {}".format(image_save_prefix))
            print("Could not find the enter/exit points, check the mask")
            raise

        # This is all just in case the wire doubles back on itself in the right or left columns
        left_side_range = max(left_side_masked_rows) - min(left_side_masked_rows)
        right_side_range = max(right_side_masked_rows) - min(right_side_masked_rows)
        # if the range is greater than 70, remove the rows that are in the top segment
        if left_side_range > 70:
            left_side_masked_rows = [i for i in left_side_masked_rows if i > min(left_side_masked_rows) + 70]
        if right_side_range > 70:
            right_side_masked_rows = [i for i in right_side_masked_rows if i > min(right_side_masked_rows) + 70]

        # Calculate the exit points as the average of the masked rows on the right and left sides
        right_exit = int(np.mean(right_side_masked_rows))
        left_exit = int(np.mean(left_side_masked_rows))

        # draw a dash at the right and left exit points on the image
        cv2.line(image_bgr,
                 (self.gripper_right_column, right_exit),
                 (self.gripper_right_column + 10, right_exit),
                 (0, 255, 0),
                 2)
        cv2.line(image_bgr,
                 (self.gripper_left_column, left_exit),
                 (self.gripper_left_column - 10, left_exit),
                 (0, 255, 0),
                 2)

        if self.viz:
            # Draw a line on the right and left sides of the gripper, and at the bottom of the gripper
            cv2.line(image_bgr,
                     (self.gripper_right_column, 0),
                     (self.gripper_right_column, image_bgr.shape[0]),
                     (0, 0, 255),
                     1)
            cv2.line(image_bgr,
                     (self.gripper_left_column, 0),
                     (self.gripper_left_column, image_bgr.shape[0]),
                     (0, 0, 255),
                     1)
            cv2.line(image_bgr,
                     (0, self.gripper_bottom_row),
                     (image_bgr.shape[1], self.gripper_bottom_row),
                     (0, 0, 255),
                     1)
            if image_save_prefix is not None:
                print("image: {}".format(image_save_prefix))
            cv2.imshow("image", image_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if image_save_prefix is None and self.captioned_image_save_directory is not None:
            raise ValueError("image_trial_num must be specified for skeleton image to be saved")
        elif self.captioned_image_save_directory is not None:
            cv2.imwrite(os.path.join(self.captioned_image_save_directory, image_save_prefix + "_enter_exit.png"),
                        image_bgr)

        # Normalize the exit points to be between 0 and 1, where 0 is the bottom of the gripper and 1 is the top of the
        # image
        right_exit = (self.gripper_bottom_row - right_exit) / self.gripper_bottom_row
        left_exit = (self.gripper_bottom_row - left_exit) / self.gripper_bottom_row

        return right_exit, left_exit

def get_ground_truth_data(segmentation_method='radius'):
    """ Gets the ground truth data for the radius and enter/exit points of the wire and saves it to a numpy array."""

    # Set this to the trial number to start at, if you want to start at the beginning set it to 0
    start_image_num = 0

    # Get the dataset info
    dataset_info = get_dataset_info()

    if segmentation_method == 'radius':
        num_gt_results = 3
    if segmentation_method == 'poly':
        num_gt_results = 5

    ground_truth_data = np.zeros((dataset_info['total_num_samples'], num_gt_results))

    ground_truther = GroundTruther(viz=True,
                                   overwrite=True)

    sample_counter = 0
    num_samples_per_trial = dataset_info['num_samples_per_trial']

    for i, image_file_name in enumerate(dataset_info['image_names']):
        image_trial_num = int(image_file_name.split('.')[0].split('_')[0])

        if image_trial_num < start_image_num:
            continue

        image_path = os.path.join(dataset_info['image_directory'], image_file_name)

        image = cv2.imread(image_path)

        if i % (num_samples_per_trial+1) == 0:
            skeleton = ground_truther.get_skeleton(image.copy(), image_trial_num)

            if segmentation_method == 'radius':
                radius = ground_truther.get_radius(image.copy(), skeleton, image_trial_num)
            if segmentation_method == 'poly':
                poly_coeffs = ground_truther.get_poly(image.copy(), skeleton, image_trial_num)   
            
            sample_counter = 0
        else:
            image_save_prefix = image_file_name.split('.')[0]
            right_exit, left_exit = ground_truther.get_enter_exit_point(image.copy(), image_save_prefix)
            trial_num_corrected = image_trial_num - len([i for i in dataset_info['skip_trials'] if i < image_trial_num])
            idx = trial_num_corrected * 10 + sample_counter
            if segmentation_method == 'radius':
                ground_truth_data[idx, 0] = radius
                ground_truth_data[idx, 1] = left_exit
                ground_truth_data[idx, 2] = right_exit
            if segmentation_method == 'poly':
                ground_truth_data[idx, 0] = poly_coeffs[0]
                ground_truth_data[idx, 1] = poly_coeffs[1]
                ground_truth_data[idx, 2] = poly_coeffs[2]
                ground_truth_data[idx, 3] = left_exit
                ground_truth_data[idx, 4] = right_exit
            sample_counter += 1

    np.save(dataset_info["ground_truth_data_path"], ground_truth_data)

    # also save as csv, but add file names
    grip_data_file_names = np.array(dataset_info['gripper_file_names'])
    grip_data_file_names = np.array([file_name.split('.')[0] for file_name in grip_data_file_names])
    ground_truth_data = np.hstack((grip_data_file_names.reshape(-1, 1), ground_truth_data))
    if segmentation_method == 'radius':
        ground_truth_data = pd.DataFrame(ground_truth_data, columns=['file_name', 'radius', 'left_exit', 'right_exit'])
    if segmentation_method == 'poly':
        ground_truth_data = pd.DataFrame(ground_truth_data, columns=['file_name', 'poly_coeff_1', 'poly_coeff_2', 'poly_coeff_3', 'left_exit', 'right_exit'])
    csv_file_path = dataset_info["ground_truth_data_path"].split('.')[0] + '.csv'
    ground_truth_data.to_csv(csv_file_path, index=False)



def setup_train_and_test_data():
    """ Gets the gripper data and saves it to a numpy array."""
    dataset_info = get_dataset_info()

    X_train = np.zeros((dataset_info['total_num_samples'], dataset_info['time_steps_to_use'], dataset_info['num_features']))

    # Loop over each training example
    for i, file_name in enumerate(dataset_info['gripper_file_names']):

        file_path = os.path.join(dataset_info['gripper_data_directory'], file_name)

        gripper_data = pd.read_csv(file_path)

        # Extract the last 30 rows and drop non-feature columns if necessary
        data = gripper_data.iloc[-dataset_info['time_steps_to_use']:].drop(columns=dataset_info['non_feature_columns']).values

        X_train[i] = data

    # normalize the training data
    og_shape = X_train.shape
    reshaped_data = X_train.reshape(-1, X_train.shape[-1])
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(reshaped_data)
    X_train = normalized_data.reshape(og_shape)

    # get the ground truth data
    y_train = np.load(dataset_info["ground_truth_data_path"])

    num_test_samples = int(dataset_info['total_num_samples'] * dataset_info['train_val_test_split'][2])

    # set the random seed if necessary
    if dataset_info["random_seed"] is not None:
        np.random.seed(dataset_info["random_seed"])

    # remove 5 random samples from X_train and y_train for testing
    test_indices = np.random.choice(range(dataset_info["total_num_samples"]), size=num_test_samples, replace=False)
    X_test = X_train[test_indices]
    y_test = y_train[test_indices]
    X_train = np.delete(X_train, test_indices, axis=0)
    y_train = np.delete(y_train, test_indices, axis=0)
    file_names_test = np.array(dataset_info['gripper_file_names'])[test_indices]

    # remove .csv from file names
    file_names_test = np.array([file_name.split('.')[0] for file_name in file_names_test])

    np.save(dataset_info["X_train_path"], X_train)
    np.save(dataset_info["y_train_path"], y_train)
    np.save(dataset_info["X_test_path"], X_test)
    np.save(dataset_info["y_test_path"], y_test)
    np.save(dataset_info["test_data_file_names_path"], file_names_test)

if __name__ == "__main__":
    get_ground_truth_data(segmentation_method='poly')
    setup_train_and_test_data()


