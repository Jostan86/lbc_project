import rospy
import numpy as np
from dataset_info import get_dataset_info
from lbc_project.srv import MoveData, MoveDataRequest
import pandas as pd

rospy.init_node("move_test")

server_connection = rospy.ServiceProxy('gripper_move_data', MoveData)

dataset_info = get_dataset_info()

num_features = dataset_info["num_features"]
num_time_steps = dataset_info["time_steps_to_use"]

x_data = np.load(dataset_info["X_test_path"])
y_data = np.load(dataset_info["y_test_path"])
file_paths = np.load(dataset_info["test_data_file_names_path"])
print(num_features * num_time_steps)
test_num = 1
x_data = x_data[test_num].reshape(num_time_steps * num_features)
y_data = y_data[test_num]
file_path = file_paths[test_num]

path_to_data = "/home/jostan/Documents/lbc_datasets/lbc_dataset4/grip_data/0_0.csv"

gripper_data = pd.read_csv(path_to_data)

# Extract the last 30 rows and drop non-feature columns if necessary
x_data = gripper_data.iloc[-dataset_info['time_steps_to_use']:].drop(
    columns=dataset_info['non_feature_columns']).values

x_data = x_data.reshape(num_time_steps * num_features)
x_data = x_data.astype(np.float32).tolist()
request = MoveDataRequest()
# request.gripper_data = x_data
request.gripper_data = x_data
request.move_right = True

response = server_connection(request)

print("Actual: {}".format(y_data))
# print("Predicted: {}".format(response.move_data))
print("response: x: {}, y: {}, angle: {}".format(response.x, response.y, response.angle))
print("File path: {}".format(file_path))



