import os

def get_dataset_info():
    """ Set all the dataset info in this function, such as the directories and file names for the dataset, and many
    other parameters.

    Returns:
        dataset_info (dict): dictionary with all the needed directories and file names for the dataset,
        and other dataset info
    """
    # put all the dataset info in a dictionary
    dataset_info = {}

    dataset_directory = "/home/jostan/Documents/lbc_datasets/lbc_dataset4"
    package_directory = "/home/jostan/catkin_ws/src/pkgs_noetic/course_pkgs/lbc/lbc_project"
    dataset_info["operation_data_path"] = os.path.join(dataset_directory, "grip_data", "0_0.csv")
    dataset_info["operation_image_path"] = os.path.join(dataset_directory, "images", "0_0.jpg")
    dataset_info["model_path"] = os.path.join(package_directory, "model_data", "model_5_2.h5")
    dataset_info["scaler_path"] = os.path.join(package_directory, "model_data", "scaler.pkl")
    dataset_info["move_distance"] = 10  # mm
    dataset_info["min_radius"] = 40  # mm

    # Trials numbers to not use for training, mostly because they don't have valid entry/exit points
    skip_trials = [32, 34, 30, 38, 39]

    dataset_info["enter_exit_multiplier"] = 10
    dataset_info["time_steps_to_use"] = 30
    dataset_info["num_epochs"] = 200
    dataset_info["batch_size"] = 10
    dataset_info["model_version"] = 1
    dataset_info["num_testing_trials"] = 6
    dataset_info["activation_function"] = "relu"
    dataset_info["loss_function"] = "mean_absolute_error"
    dataset_info["optimizer"] = "adam"
    dataset_info["num_units"] = 50

    num_trials = 80
    num_samples_per_trial = 10
    num_features = 139

    # Set features to not use here
    non_feature_columns2 = ['0_slipstate_{}'.format(i) for i in range(8)]
    non_feature_columns4 = ['1_slipstate_{}'.format(i) for i in range(8)]
    non_feature_columns5 = ['0_friction_est', '1_friction_est', '0_target_grip_force', '0_is_ref_loaded',
                            '0_is_sd_active', '1_target_grip_force', '1_is_ref_loaded',
                            '1_is_sd_active',
                            'gripper_pos'
                            ]

    non_feature_columns = non_feature_columns2 + non_feature_columns4 + non_feature_columns5
    # non_feature_columns = []

    num_features -= len(non_feature_columns)

    # construct list of image paths
    image_names = []
    for i in range(num_trials):
        if i in skip_trials:
            continue
        image_names.append('{}.jpg'.format(i))
        for j in range(num_samples_per_trial):
            image_names.append('{}_{}.jpg'.format(i, j))

    gripper_file_names = []
    for i in range(num_trials):
        if i in skip_trials:
            continue
        for j in range(num_samples_per_trial):
            gripper_file_names.append('{}_{}.csv'.format(i, j))


    # Set all the directories
    dataset_info["dataset_directory"] = dataset_directory
    dataset_info["package_directory"] = package_directory
    dataset_info["image_directory"] = os.path.join(dataset_directory, "images")
    dataset_info["gripper_data_directory"] = os.path.join(dataset_directory, "grip_data")
    dataset_info["annotated_image_directory"] = os.path.join(dataset_directory, "annotated_images")
    dataset_info["trials_directory"] = os.path.join(dataset_directory, "trials")
    dataset_info["ground_truth_data_path"] = os.path.join(dataset_directory, "ground_truth_data.npy")
    dataset_info["X_train_path"] = os.path.join(dataset_directory, "X_train.npy")
    dataset_info["y_train_path"] = os.path.join(dataset_directory, "y_train.npy")
    dataset_info["X_val_path"] = os.path.join(dataset_directory, "X_val.npy")
    dataset_info["y_val_path"] = os.path.join(dataset_directory, "y_val.npy")
    dataset_info["X_test_path"] = os.path.join(dataset_directory, "X_test.npy")
    dataset_info["y_test_path"] = os.path.join(dataset_directory, "y_test.npy")
    dataset_info["test_data_file_names_path"] = os.path.join(dataset_directory, "test_data_file_names.npy")
    dataset_info["val_data_file_names_path"] = os.path.join(dataset_directory, "val_data_file_names.npy")
    dataset_info["train_data_file_names_path"] = os.path.join(dataset_directory, "train_data_file_names.npy")

    # Lists of image and gripper file names that will be used for training
    dataset_info["image_names"] = image_names
    dataset_info["gripper_file_names"] = gripper_file_names

    # Trials to not use for training, mostly because they don't have valid entry/exit points
    dataset_info["skip_trials"] = skip_trials

    # Other dataset parameters / info
    dataset_info["num_trials_used"] = num_trials - len(skip_trials)
    dataset_info["num_samples_per_trial"] = num_samples_per_trial
    dataset_info["total_num_samples"] = dataset_info["num_trials_used"] * num_samples_per_trial
    dataset_info["non_feature_columns"] = non_feature_columns
    dataset_info["num_features"] = num_features

    dataset_info["train_val_test_split"] = [0.90, 0.1, 0]

    # Max radius to use (in pixels), values greater than this will be set to this value
    dataset_info["max_radius"] = 500
    # Column of the image that has the center of the gripper
    dataset_info["center_of_gripper_pixel"] = 292
    # Number of pixels to keep on either side of the center of the gripper for various crops
    dataset_info["center_segment_half_width"] = 70
    # Number of pixels to crop from the top and bottom of the image (used for the skeleton)
    dataset_info["bottom_crop"] = 100
    dataset_info["top_crop"] = 50
    # Columns and rows of the image that are the left, right and bottom of the gripper, used for enter/exit point calculation
    dataset_info["gripper_right_column"] = 351
    dataset_info["gripper_left_column"] = 228
    dataset_info["gripper_bottom_row"] = 450
    dataset_info["pixels_per_mm"] = 4.05
    dataset_info["gripper_center_row"] = 348 #375

    # Seed to use to keep consistent results, can be set to None to not set a seed
    dataset_info["random_seed"] = 42

    return dataset_info

def image_to_gripper_transform(pixel_row, pixel_column, dataset_info):
    """ Transform pixel coordinates to gripper coordinates

    Args:
        pixel_row (int): row of the pixel
        pixel_column (int): column of the pixel
        dataset_info (dict): dictionary with all the needed directories and file names for the dataset,
        and other dataset info

    Returns:
        gripper_coords (tuple): gripper coordinates of the form (x, y) in mm
    """
    # get the gripper coordinates
    gripper_center_row = dataset_info["gripper_center_row"]
    gripper_center_column = dataset_info["center_of_gripper_pixel"]

    # get the pixels per mm
    pixels_per_mm = dataset_info["pixels_per_mm"]

    # get the gripper coordinates
    x_relative = (pixel_column - gripper_center_column) / pixels_per_mm
    y_relative = (-(pixel_row - gripper_center_row)) / pixels_per_mm
    return x_relative, y_relative

def gripper_to_image_transform(x_relative, y_relative, dataset_info):
    """ Transform gripper coordinates to pixel coordinates

    Args:
        x_relative (float): x coordinate of the gripper in mm
        y_relative (float): y coordinate of the gripper in mm
        dataset_info (dict): dictionary with all the needed directories and file names for the dataset,
        and other dataset info

    Returns:
        pixel_coords (tuple): pixel coordinates of the form (row, column)
    """
    # get the gripper coordinates
    gripper_center_row = dataset_info["gripper_center_row"]
    gripper_center_column = dataset_info["center_of_gripper_pixel"]

    # get the pixels per mm
    pixels_per_mm = dataset_info["pixels_per_mm"]

    # get the gripper coordinates
    pixel_column = x_relative * pixels_per_mm + gripper_center_column
    pixel_row = -(y_relative * pixels_per_mm - gripper_center_row)
    return pixel_row, pixel_column

if __name__ == "__main__":
    import numpy as np

    dataset_info = get_dataset_info()
    rows = (100, 100, 500, 500)
    columns = (100, 500, 100, 500)
    xy_coords = ((-47.41, 67.9), (51.36, 67.9), (-47.41, -30.86), (51.36, -30.86))
    pass_test = True
    for row, column, xy_coord in zip(rows, columns, xy_coords):
        result = image_to_gripper_transform(row, column, dataset_info)
        try:
            assert np.allclose(result, xy_coord, atol=0.1)
        except AssertionError:
            print("AssertionError")
            print(result)
            print(xy_coord)
            print(np.allclose(result, xy_coord, atol=0.1))
            pass_test = False


    for row, column, xy_coord in zip(rows, columns, xy_coords):
        result = gripper_to_image_transform(xy_coord[0], xy_coord[1], dataset_info)
        try:
            assert np.allclose(result, (row, column), atol=0.1)
        except AssertionError:
            print("AssertionError")
            print(result)
            print(xy_coord)
            print(np.allclose(result, xy_coord, atol=0.1))
            pass_test = False
    if pass_test:
        print("All tests passed")