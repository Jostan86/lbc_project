import keras.models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
from process_data import get_dataset_info
import cv2
import math
import os
from process_results import ResultsProcessor
from process_data import setup_train_and_val_data
import json
import csv
import pandas as pd

def train_model(num_epochs, dataset_info):
    """ Trains the LSTM model and saves it. Also plots the training and validation loss."""


    X_train = np.load(dataset_info["X_train_path"])
    y_train = np.load(dataset_info["y_train_path"])
    X_val = np.load(dataset_info["X_val_path"])
    y_val = np.load(dataset_info["y_val_path"])

    # Define the LSTM model with multiple layers
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(dataset_info["num_units"],
                   activation=dataset_info["activation_function"],
                   input_shape=(dataset_info["time_steps_to_use"], dataset_info["num_features"]),
                   # return_sequences=False,
                   # recurrent_dropout=0.2
                   ))

    model.add(Dense(3))  # Output layer: Predicts the radius

    model.compile(optimizer=dataset_info["optimizer"], loss=dataset_info["loss_function"])

    # Assuming X_train and y_train are your input and output training data
    history = model.fit(X_train,
                        y_train,
                        epochs=num_epochs,
                        validation_data=(X_val, y_val),
                        batch_size=dataset_info["batch_size"]
                        )

    # Plot accuracy from training history
    plt.plot(history.history['loss'], label='Training Loss')
    # check if validation loss is in history
    if "val_loss" in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # save the model
    model.save(dataset_info["model_path"])
    # return the model and the plot and the final training and validation loss
    return model, plt, history.history['loss'], history.history['val_loss']

def test_model(model=None, overwrite=False, results_save_path=None, verbose=False):
    """ Tests the model on the test data and saves the results as annotated images.

    Args:
        model (tensorflow.keras.models.Sequential): The model to test. If None, the model will be loaded from the dataset directory.
        overwrite (bool): If True, the results directory will be overwritten. If False, an error will be raised if the results directory is not empty.
    """

    dataset_info = get_dataset_info()
    results_processor = ResultsProcessor(model)

    if results_save_path is None:
        results_save_path = os.path.join(dataset_info["dataset_directory"], "results2")

        if not os.path.exists(results_save_path):
            raise ValueError("The results directory does not exist.")

        if not overwrite and os.listdir(results_save_path) != []:
            raise ValueError("The results directory is not empty, set overwrite to True to overwrite the directory.")
        else:
            for file in os.listdir(results_save_path):
                os.remove(os.path.join(results_save_path, file))


    X_test = np.load(dataset_info["X_val_path"])
    y_test = np.load(dataset_info["y_val_path"])
    file_names_test = np.load(dataset_info["val_data_file_names_path"])
    # X_test = np.load(dataset_info["X_train_path"])
    # y_test = np.load(dataset_info["y_train_path"])
    # file_names_test = np.load(dataset_info["train_data_file_names_path"])
    total_distance_error = 0
    total_angle_error = 0

    predictions = results_processor.get_result(X_test, transform=False)

    for i, (y_test, prediction, image_name) in enumerate(zip(y_test, predictions, file_names_test)):
        if verbose:
            print("Image:", image_name)
        image_path = os.path.join(dataset_info["image_directory"], image_name + ".jpg")
        image = cv2.imread(image_path)


        move_xy_right, move_angle_right, image = results_processor.process_result(prediction, image, move_right=True, color=(57, 127, 255))
        move_xy_right_gt, move_angle_right_gt, image = results_processor.process_result(y_test, image, move_right=True, color=(255, 204, 52), ground_truth=True)
        move_xy_left, move_angle_left, image = results_processor.process_result(prediction, image, move_right=False, color=(57, 127, 255))
        move_xy_left_gt, move_angle_left_gt, image = results_processor.process_result(y_test, image, move_right=False, color=(255, 204, 52), ground_truth=True)

        # get the sum of difference between the predicted and ground truth
        dist_right = np.linalg.norm(np.asarray(move_xy_right) - np.asarray(move_xy_right_gt))
        dist_left = np.linalg.norm(np.asarray(move_xy_left) - np.asarray(move_xy_left_gt))

        # print the total
        total_distance_error += dist_right + dist_left
        total_angle_error += abs(move_angle_right - move_angle_right_gt) + abs(move_angle_left - move_angle_left_gt)

        # save the image
        cv2.imwrite(os.path.join(results_save_path, "result_{}.png".format(image_name.split(".")[0])), image)

        if verbose:
            print("Total distance error: {}".format(dist_right + dist_left))
            print("Total angle error: {}".format(abs(move_angle_right - move_angle_right_gt) + abs(move_angle_left - move_angle_left_gt)))

            # Set numpy to print 2 decimal places
            np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
            print("Prediction:", prediction, "Actual:", y_test)
    print("num corrections: {}".format(results_processor.counter))
    average_distance_error = total_distance_error / len(X_test)
    print("Average distance error: {}".format(average_distance_error))
    average_angle_error = total_angle_error / len(X_test)
    return average_distance_error, average_angle_error, results_processor.counter

def run_trial(note=None):

    setup_train_and_val_data()

    dataset_info = get_dataset_info()
    trials_directory = dataset_info["trials_directory"]
    # get the current trial number
    trial_number = len(os.listdir(trials_directory)) - 2

    # create the trial directory
    trial_directory = os.path.join(trials_directory, str(trial_number))
    # check if the directory already exists
    if os.path.exists(trial_directory):
        raise ValueError("The trial directory already exists.")
    os.mkdir(trial_directory)

    # save the dataset info as a json file
    dataset_info_path = os.path.join(trial_directory, "dataset_info.json")
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f)

    average_distance_errors = []
    average_angle_errors = []
    num_radius_corrections_list = []
    # save the model
    for i in range(dataset_info["num_testing_trials"]):
        # create the image directory
        image_directory = os.path.join(trial_directory, "images_{}".format(i))
        os.mkdir(image_directory)
        model_path = os.path.join(trial_directory, "model_{}.h5".format(i))
        model, plt, train_loss, val_loss = train_model(num_epochs=dataset_info["num_epochs"], dataset_info=dataset_info)
        model.save(model_path)

        # save the plot
        plot_path = os.path.join(trial_directory, "plot_{}.png".format(i))
        plt.savefig(plot_path)
        # save the training and validation loss
        loss_path = os.path.join(trial_directory, "loss_{}.csv".format(i))
        with open(loss_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["train_loss", "val_loss"])
            for train_loss, val_loss in zip(train_loss, val_loss):
                writer.writerow([train_loss, val_loss])

        # test the model
        average_distance_error, average_angle_error, num_radius_corrections = test_model(model=model, results_save_path=image_directory, verbose=False)
        average_distance_errors.append(average_distance_error)
        average_angle_errors.append(average_angle_error)
        num_radius_corrections_list.append(num_radius_corrections)

    average_distance_error = sum(average_distance_errors) / len(average_distance_errors)
    average_angle_error = sum(average_angle_errors) / len(average_angle_errors)
    num_radius_corrections = sum(num_radius_corrections_list) / len(num_radius_corrections_list)

    # save a copy of the code used
    code_directory = os.path.join(trial_directory, "code")
    os.mkdir(code_directory)
    file_names = ["dataset_info.py", "process_data.py", "process_results.py", "train_test_model.py", "scaler.pkl"]

    for file_name in file_names:
        package_directory = dataset_info["package_directory"]
        file_name = os.path.join(package_directory, "src", file_name)
        os.system("cp {} {}".format(file_name, code_directory))

    results_path = os.path.join(trials_directory, "all_results.csv")

    data_to_add = {"trial_number": [trial_number],
                   "average_distance_error": [average_distance_error],
                   "distance_errors": [average_distance_errors],
                   "average_angle_error": [average_angle_error],
                   "num_trials": [dataset_info["num_testing_trials"]],
                   "num_epochs": [dataset_info["num_epochs"]],
                   "batch_size": [dataset_info["batch_size"]],
                   "model_version": [dataset_info["model_version"]],
                   "enter_exit_multiplier": [dataset_info["enter_exit_multiplier"]],
                   "num_time_steps_used": [dataset_info["time_steps_to_use"]],
                   "num_features": [dataset_info["num_features"]],
                   "num_units": [dataset_info["num_units"]],
                   "loss_function": [dataset_info["loss_function"]],
                   "optimizer": [dataset_info["optimizer"]],
                   "activation_function": [dataset_info["activation_function"]],
                   "removed_features": [dataset_info["non_feature_columns"]],
                   "train_val_test_split": [dataset_info["train_val_test_split"]],
                   "min_radius": [dataset_info["min_radius"]],
                   "max_radius": [dataset_info["max_radius"]],
                   "move_distance": [dataset_info["move_distance"]],
                   "num_min_radius_corrections": [num_radius_corrections],
                    }

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df = pd.concat([df, pd.DataFrame.from_dict(data_to_add)], ignore_index=True)

    else:
        df = pd.DataFrame.from_dict(data_to_add)

    df.to_csv(results_path, index=False)

    # add a note about the trial
    if note is None:
        note = input("Enter a note about the trial: ")

    # check if new note is same as previous note
    previous_note = None
    if trial_number > 0:
        prev_note_path = os.path.join(trials_directory, "notes", "note_{}.txt".format(trial_number - 1))
        if os.path.exists(prev_note_path):
            with open(os.path.join(trials_directory, "notes", "note_{}.txt".format(trial_number - 1)), "r") as f:
                previous_note = f.read()
    if note == previous_note:
        note = input("Note is the same as previous note, enter a new note: ")

    with open(os.path.join(trials_directory, "notes", "note_{}.txt".format(trial_number)), "w") as f:
        f.write(note)

    print("Average distance errors: {}".format(average_distance_errors))
    print("Average distance error: {}".format(average_distance_error))




if __name__ == "__main__":
    # model = train_model(num_epochs=100, run_setup=True)
    # test_model(overwrite=True)
    note = "trying a final model with most of the data"
    run_trial(note)
    # model_path = "../model_data/model_29_0.h5"
    # model = keras.models.load_model(model_path)
    # test_model(model=model, verbose=True)



