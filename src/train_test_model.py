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

class CustomEvaluationCallback(tf.keras.callbacks.Callback):
    """ Custom callback to print the loss after each epoch.

    """
    def __init__(self, evaluation_data, dataset_info, model, data_save_directory=None, trial_num=None):
        super(CustomEvaluationCallback, self).__init__()
        self.dataset_info = dataset_info
        self.model = model
        self.data_save_directory = data_save_directory
        self.trial_num = trial_num
        self.X_val = evaluation_data[0]
        self.y_val = evaluation_data[1]
        self.results_processor = ResultsProcessor(model)
        self.stats = {"average_distance_error": [],
                        "average_angle_error": [],
                        "median_distance_error": [],
                        "distance_error_std": [],
                        "num_radius_corrections": []}


    def on_epoch_end(self, epoch, logs=None):
        stats = test_model(model=self.model, dataset_info=self.dataset_info, results_processor=self.results_processor)
        self.stats["average_distance_error"].append(stats["average_distance_error"])
        self.stats["average_angle_error"].append(stats["average_angle_error"])
        self.stats["median_distance_error"].append(stats["median_distance_error"])
        self.stats["distance_error_std"].append(stats["distance_error_std"])
        self.stats["num_radius_corrections"].append(stats["num_radius_corrections"])
        print("Average distance error: {}".format(stats["average_distance_error"]))

    def on_train_end(self, logs=None):
        if self.data_save_directory is None:
            return

        if self.trial_num is None:
            stats_path = os.path.join(self.data_save_directory, "stats.csv")
            plot_path = os.path.join(self.data_save_directory, "distance_error_plot.png")
        else:
            stats_path = os.path.join(self.data_save_directory, "stats_{}.csv".format(self.trial_num))
            plot_path = os.path.join(self.data_save_directory, "distance_error_plot_{}.png".format(self.trial_num))
        with open(stats_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["average_distance_error", "average_angle_error", "median_distance_error", "distance_error_std", "num_radius_corrections"])
            for average_distance_error, average_angle_error, median_distance_error, distance_error_std, num_radius_corrections in zip(self.stats["average_distance_error"], self.stats["average_angle_error"], self.stats["median_distance_error"], self.stats["distance_error_std"], self.stats["num_radius_corrections"]):
                writer.writerow([average_distance_error, average_angle_error, median_distance_error, distance_error_std, num_radius_corrections])

        # make a plot of the average distance error, and median distance error
        plt.plot(self.stats["average_distance_error"], label="Average Distance Error")
        # plt.plot(self.stats["median_distance_error"], label="Median Distance Error")
        plt.xlabel("Epochs")
        plt.ylabel("Distance Error (mm)")
        plt.title("Average Distance Error")
        plt.savefig(plot_path)
        plt.show()
        # clear the plot
        plt.clf()

def train_model(dataset_info, data_save_directory=None, trial_num=None, evaluate_with_metric=True):
    """ Trains the LSTM model and saves it. Also plots the training and validation loss.

    Args:
        dataset_info (dict): A dictionary containing the dataset info.
        data_save_directory (str): The directory to save the training plot and loss csv file. If None, the plot and csv file will not be saved.
        trial_num (int): The trial number. Adds the trial number to the plot and csv file names. If None, the trial number will not be added to the plot and csv file names.
evaluate_with_metric (bool): If True, the evaluation metric will be run each epoch

    """
    X_train = np.load(dataset_info["X_train_path"])
    y_train = np.load(dataset_info["y_train_path"])
    X_val = np.load(dataset_info["X_val_path"])
    y_val = np.load(dataset_info["y_val_path"])

    model = Sequential()

    # First LSTM layer
    model.add(LSTM(dataset_info["num_units"],
                   activation=dataset_info["activation_function"],
                   input_shape=(dataset_info["time_steps_to_use"], dataset_info["num_features"]),
                   # return_sequences=False,
                   # recurrent_dropout=0.2
                   ))

    model.add(Dense(3))

    model.compile(optimizer=dataset_info["optimizer"], loss=dataset_info["loss_function"])

    if evaluate_with_metric:
        custom_evaluation_callback = CustomEvaluationCallback(evaluation_data=(X_val, y_val), dataset_info=dataset_info,
                                                              model=model, data_save_directory=data_save_directory,
                                                              trial_num=trial_num)
        history = model.fit(X_train,
                            y_train,
                            epochs=dataset_info["num_epochs"],
                            validation_data=(X_val, y_val),
                            batch_size=dataset_info["batch_size"],
                            callbacks=[custom_evaluation_callback]
                            )
    else:
        history = model.fit(X_train,
                            y_train,
                            epochs=dataset_info["num_epochs"],
                            validation_data=(X_val, y_val),
                            batch_size=dataset_info["batch_size"],
                            )

    # Plot accuracy from training history
    plt.plot(history.history['loss'], label='Training Loss')
    # check if validation loss is in history
    if "val_loss" in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MAE Loss')
    plt.legend()

    # save the model
    model.save(dataset_info["model_path"])

    if data_save_directory is not None:
        if trial_num is None:
            plot_path = os.path.join(data_save_directory, "plot.png")
            loss_path = os.path.join(data_save_directory, "loss.csv")
        else:
            plot_path = os.path.join(data_save_directory, "plot_{}.png".format(trial_num))
            loss_path = os.path.join(data_save_directory, "loss_{}.csv".format(trial_num))

        plt.savefig(plot_path)


        with open(loss_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["train_loss", "val_loss"])
            for train_loss, val_loss in zip(history.history['loss'], history.history['val_loss']):
                writer.writerow([train_loss, val_loss])
    plt.show()
    return model

def test_model(model=None, dataset_info=None, results_processor=None, overwrite=False, image_save_directory=None, verbose=False):
    """ Tests the model on the test data and saves the results as annotated images.

    Args:
        model (tensorflow.keras.models.Sequential): The model to test. If None, the model will be loaded.
        dataset_info (dict): A dictionary containing the dataset info. If None, the dataset info will be loaded from the dataset directory.
        results_processor (ResultsProcessor): The results processor to use. If None, a new results processor will be created.
        overwrite (bool): If True, the results directory will be overwritten. If False, an error will be raised if the results directory is not empty.
        image_save_directory (str): The directory to save the images. If None, the images will not be saved.
        verbose (bool): If True, the results will be printed.

    Returns:
        average_distance_error (float): The average distance error.
        average_angle_error (float): The average angle error.
        num_radius_corrections (int): The number of times the radius was corrected.

    """

    if dataset_info is None:
        dataset_info = get_dataset_info()
    if model is None:
        model = keras.models.load_model(dataset_info["model_path"])
    if results_processor is None:
        results_processor = ResultsProcessor(model)

    if image_save_directory is not None:
        if not os.path.exists(image_save_directory):
            raise ValueError("The results directory does not exist.")
        else:
            if not overwrite and os.listdir(image_save_directory) != []:
                raise ValueError("The results directory is not empty, set overwrite to True to overwrite the directory.")
            else:
                for file in os.listdir(image_save_directory):
                    os.remove(os.path.join(image_save_directory, file))

    X_test = np.load(dataset_info["X_val_path"])
    y_test = np.load(dataset_info["y_val_path"])
    file_names_test = np.load(dataset_info["val_data_file_names_path"])

    predictions = results_processor.get_result(X_test, transform=False)

    distance_errors = []
    angle_errors = []

    for i, (y_test, prediction, image_name) in enumerate(zip(y_test, predictions, file_names_test)):
        if verbose:
            print("Sample num:", image_name)

        if image_save_directory is not None:
            image_path = os.path.join(dataset_info["image_directory"], image_name + ".jpg")
            image = cv2.imread(image_path)

            gt_color = (110, 245, 73)
            pred_color = (5, 158, 245)
            move_xy_right, move_angle_right, image = results_processor.process_result(prediction, image, move_right=True, color=pred_color)
            move_xy_right_gt, move_angle_right_gt, image = results_processor.process_result(y_test, image, move_right=True, color=gt_color, ground_truth=True)
            move_xy_left, move_angle_left, image = results_processor.process_result(prediction, image, move_right=False, color=pred_color)
            move_xy_left_gt, move_angle_left_gt, image = results_processor.process_result(y_test, image, move_right=False, color=gt_color, ground_truth=True)
            # save the image
            cv2.imwrite(os.path.join(image_save_directory, "result_{}.png".format(image_name.split(".")[0])), image)

        else:
            move_xy_right, move_angle_right, _ = results_processor.process_result(prediction, move_right=True)
            move_xy_right_gt, move_angle_right_gt, _ = results_processor.process_result(y_test, move_right=True, ground_truth=True)
            move_xy_left, move_angle_left, _ = results_processor.process_result(prediction, move_right=False)
            move_xy_left_gt, move_angle_left_gt, _ = results_processor.process_result(y_test, move_right=False, ground_truth=True)

        # get the sum of difference between the predicted and ground truth
        dist_right = np.linalg.norm(np.asarray(move_xy_right) - np.asarray(move_xy_right_gt))
        dist_left = np.linalg.norm(np.asarray(move_xy_left) - np.asarray(move_xy_left_gt))

        distance_errors.append(dist_right + dist_left)
        angle_errors.append(abs(move_angle_right - move_angle_right_gt) + abs(move_angle_left - move_angle_left_gt))

        if verbose:
            print("Total distance error: {}".format(dist_right + dist_left))
            print("Total angle error: {}".format(abs(move_angle_right - move_angle_right_gt) + abs(move_angle_left - move_angle_left_gt)))

            # Set numpy to print 2 decimal places
            np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
            print("Prediction:", prediction, "Actual:", y_test)


    num_corrections = results_processor.counter
    average_distance_error = sum(distance_errors) / len(distance_errors)
    average_angle_error = sum(angle_errors) / len(angle_errors)
    distance_error_std = np.std(np.asarray(distance_errors))
    distance_errors.sort()
    median_distance_error = distance_errors[len(distance_errors) // 2]

    test_stats = {"average_distance_error": average_distance_error,
                  "average_angle_error": average_angle_error,
                  "median_distance_error": median_distance_error,
                  "distance_error_std": distance_error_std,
                  "num_radius_corrections": num_corrections}
    return test_stats

def run_trial(note=None):
    """ Runs a trial. Saves the model, results, and dataset info. Parameters are set in dataset_info.py.

    Args:
        note (str): A note about the trial. If None, the user will be prompted to enter a note.

    """

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
        model, plt, train_loss, val_loss = train_model(dataset_info=dataset_info,
                                                       data_save_directory=trial_directory,
                                                       trial_num=i)
        model.save(model_path)

        # test the model
        stats = test_model(model=model, image_save_directory=image_directory, verbose=False)
        average_distance_errors.append(stats["average_distance_error"])
        average_angle_errors.append(stats["average_angle_error"])
        num_radius_corrections_list.append(stats["num_radius_corrections"])

    average_distance_error = sum(average_distance_errors) / len(average_distance_errors)
    average_angle_error = sum(average_angle_errors) / len(average_angle_errors)
    num_radius_corrections = sum(num_radius_corrections_list) / len(num_radius_corrections_list)

    # save a copy of the code used
    code_directory = os.path.join(trial_directory, "code")
    os.mkdir(code_directory)
    file_names = ["dataset_info.py", "process_data.py", "process_results.py", "train_test_model.py", "../model_data/scaler.pkl"]
    package_directory = dataset_info["package_directory"]
    for file_name in file_names:
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

    # dataset_info = get_dataset_info()
    # model = keras.models.load_model(dataset_info["model_path"])

    # note = "testing multiplier effect on mae2"
    # run_trial(note)

    model = train_model(dataset_info=get_dataset_info(), data_save_directory="/home/jostan/Documents/trials1", evaluate_with_metric=True)


    # image_save_directory = "/home/jostan/Documents/trials1/images"
    # test_model(image_save_directory=image_save_directory, verbose=True)



