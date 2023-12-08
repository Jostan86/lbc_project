import rospy
import numpy as np
from dataset_info import get_dataset_info
from lbc_project.srv import MoveData, MoveDataRequest
import pandas as pd
import os

rospy.init_node("move_test")

server_connection = rospy.ServiceProxy('gripper_move_data', MoveData)

dataset_info = get_dataset_info()

num_features = dataset_info["num_features"]
num_time_steps = dataset_info["time_steps_to_use"]

# test_data_file_paths = np.load(dataset_info["val_data_file_names_path"])
test_data_file_paths = os.listdir("/media/jostan/portabits/kyle")
num_files = int(len(test_data_file_paths)/2)
for i in range(num_files):
    # file_path = test_data_file_paths[i]
    # trial_num = float(file_path.split("_")[0])images
    # sample_num = float(file_path.split("_")[1])
    # print("Trial: {}, Sample: {}".format(trial_num, sample_num))

    request = MoveDataRequest()

    # request.gripper_data = x_data
    request.gripper_data = [i, 5.9]

    request.move_right = False

    response = server_connection(request)

    # print("Actual: {}".format(y_data))
    # print("Predicted: {}".format(response.move_data))
    print("response: x: {}, y: {}, angle: {}".format(response.x, response.y, response.angle))
    # print("File path: {}".format(file_path))

    waiting = input("Press enter to continue")
    # rospy.sleep(0.2)


