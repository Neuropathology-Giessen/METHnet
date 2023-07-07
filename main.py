import csv
import enum
import string
import arguments.setting as setting
import datastructure.dataset as dataset
import os

from learning.construct_features import construct_features
from progress.bar import IncrementalBar
import learning.train
import learning.test
import getopt, sys

def run(data, setting, train=False, features_only=False, runs_start=0, runs=10, draw_map=False):
    """ Run model training and testing

    Parameters
    ----------
    data : Dataset
        Dataset to use for model training/testing
    setting : Setting
        Setting as specified by class
    train : bool
        True if want to train models
    features_only : bool
        True if only want to encode features
    runs_stat : int
        First run of Monte-Carlo cross-validation to run
    runs : int
        Last run of Monte-Carlo cross-validation to run
    draw_map : bool
        True if want to save attention maps
    """
    if runs_start >= runs:
        return 
    import numpy as np

    print(np.shape(data.train_set))
    print("_-----")
    # Create features
    construct_features(data.get_train_set(), setting)
    construct_features(data.get_validation_set(), setting)
    construct_features(data.get_test_set(), setting)

    if features_only:
        return

    bar = IncrementalBar('Running Monte Carlo ', max=runs)

    balanced_accuracies = []
    sensitivities = []
    specificities = []
    # Iterate Monte-carlo
    for k in range(runs_start, runs):
        # Set split
        data.set_fold(k)
        # Train model
        if train:
            learning.train.train(data.get_train_set(), data.get_validation_set(), k, setting)
        # Test model
        balanced_accuracy, sensitivity, specificity = learning.test.test(data.get_test_set(), k, setting, draw_map=draw_map)
        balanced_accuracies.append(balanced_accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        bar.next()

    bar.finish()
    # Save results
    for patients in data.get_test_set():
        for p in patients:
            results_folder = setting.get_data_setting().get_results_folder()
            p.save_predicted_scores(results_folder)
            if draw_map:
                p.save_map()
    
    balanced_accuracies = np.array(balanced_accuracies)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    print(np.mean(balanced_accuracies))
    print(1.96*np.std(balanced_accuracies))
    print(np.mean(sensitivities))
    print(1.96*np.std(sensitivities))
    print(np.mean(specificities))
    print(1.96*np.std(specificities))

    print(sensitivities)
    print(specificities)
    combined = np.abs(sensitivities-specificities)
    print(np.mean(combined))
    print(1.96*np.std(combined))


def run_train(data_directories, csv_file, working_directory):
    """ Set up setting and dataset and run training/testing
    """
    s = setting.Setting(data_directories, csv_file, working_directory)

    data = dataset.Dataset(s)
    
    run(data, s, train=False, features_only=False, runs_start=0,runs=s.get_network_setting().get_runs(), draw_map=True)



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hd:c:w:", ["data_directory=","csv_file=","working_directory="])
    except getopt.GetoptError:
        print('main.py -d <data_directory> -c <csv_file> -w <working_directory>')
        sys.exit(2)
    opts_vals = [o[0] for o in opts]
    if not('-d' in opts_vals or '--data_directory' in opts_vals):
        print('Specify -d or --data_directory')
        sys.exit(2)
    if not('-c' in opts_vals or '--csv_file' in opts_vals):
        print('Specify -c or --csv_file')
        sys.exit(2)
    if not('-w' in opts_vals or '--working_directory' in opts_vals):
        print('Specify -w or --working_directory')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
             print('main.py -d <data_directory> -c <csv_file> -w <working_directory>')
        elif opt in ('-d', '--data_directory'):
            data_directory = arg.strip('[]').split(',')
        elif opt in ('-c', '--csv_file'):
            if type(arg) == str and arg.endswith('.csv'):
                csv_file = arg
            else:
                print("Wrong data type for -c or --csv_file should be path to .csv")
                sys.exit(2)
        elif opt in ('-w', '--working_directiory'):
            if type(arg) == str:
                working_directory = arg
            else:
                print("Wrong data type for -w or --working_directory should be string")
                sys.exit(2)

    run_train(data_directory, csv_file, working_directory)


if __name__=="__main__":
    main(sys.argv[1:])

