import os

def get_dataset_info():
    """ Set all the dataset info in this function, such as the directories and file names for the dataset, and many
    other parameters.

    Returns:
        dataset_info (dict): dictionary with all the needed directories and file names for the dataset,
        and other dataset info
    """

    # Trials numbers to not use for training, mostly because they don't have valid entry/exit points
    skip_trials = [32, 34, 30, 38, 39]

    num_trials = 80
    num_samples_per_trial = 10
    num_features = 139

    # # Set features to not use here
    # # non_feature_columns1 = ['0_incontact_{}'.format(i) for i in range(8)]
    # non_feature_columns2 = ['0_slipstate_{}'.format(i) for i in range(8)]
    # # non_feature_columns3 = ['1_incontact_{}'.format(i) for i in range(8)]
    # non_feature_columns4 = ['1_slipstate_{}'.format(i) for i in range(8)]
    # non_feature_columns5 = ['0_friction_est', '1_friction_est', '0_target_grip_force', '0_is_ref_loaded',
    #                         '0_is_sd_active', '1_target_grip_force', '1_is_ref_loaded',
    #                         '1_is_sd_active']
    #
    # non_feature_columns = non_feature_columns2 + non_feature_columns4 + non_feature_columns5
    non_feature_columns = []

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

    # put all the dataset info in a dictionary
    dataset_info = {}

    # Set all the directories
    dataset_directory = "/home/jostan/Documents/lbc_dataset4"
    dataset_info["dataset_directory"] = dataset_directory
    dataset_info["image_directory"] = os.path.join(dataset_directory, "images")
    dataset_info["gripper_data_directory"] = os.path.join(dataset_directory, "grip_data")
    dataset_info["annotated_image_directory"] = os.path.join(dataset_directory, "annotated_images")
    dataset_info["ground_truth_data_path"] = os.path.join(dataset_directory, "ground_truth_data.npy")
    dataset_info["X_train_path"] = os.path.join(dataset_directory, "X_train.npy")
    dataset_info["y_train_path"] = os.path.join(dataset_directory, "y_train.npy")
    dataset_info["X_test_path"] = os.path.join(dataset_directory, "X_test.npy")
    dataset_info["y_test_path"] = os.path.join(dataset_directory, "y_test.npy")
    dataset_info["test_data_file_names_path"] = os.path.join(dataset_directory, "test_data_file_names.npy")
    dataset_info["model_path"] = os.path.join(dataset_directory, "lstm_model.h5")

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
    dataset_info["time_steps_to_use"] = 30
    dataset_info["train_val_test_split"] = [0.76, 0.2, 0.04]

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

    # Seed to use to keep consistent results, can be set to None to not set a seed
    dataset_info["random_seed"] = None

    return dataset_info