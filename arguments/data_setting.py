#from tkinter import Image
import utils.helper as helper
from albumentations import (RandomCrop, Normalize, Compose)

from utils.image_property import ImageProperty
from utils.tile_property import TileProperty

import os

class DataSetting(object):
    """ Class to hold parameters for data used

    Attributes
    ----------
    excluded patients : list [string]
        List of patient identifiers to manually exclude from data
    data_folders : list [string]
        Folders where WSI data is stored
    csv_patients : string
        .csv file holding patient information
    json_tiling_folder : string
        Folder where .json with Tiling information should be stored
    attention_folder : string
        Folder where .json with Attention map information should be stored
    image_properties : list [ImageProperty]
        ImageProperty Objects used to define WSIs loaded
    tile_properties : list [TileProperty]
        TileProperty Objects used to define extracted Tiles for each WSI
    skip_existing : bool
        True if want to skip existing Tilings and Features and not redo them
    split : tuple (float, float, float)
        Percentage of data to use for training/validation/test set should sum up to 1.0
    folds : list
        Folds to run if using k-fold cross-validation
    monte_carlo : bool
        True if using Monte-Carlo cross-validation instead k-fold
    monte_carlo_folder : string
        Folder where order of random shuffling in Monte-Carlo cross validation will be stored
    narrow_validation : bool
        True if want to predict only - all valid patients will be in test set
    feature_folder : string
        Folder where encoded features will be stored
    results_folder : string
        Folder where prediction results for each patient will be stored
    attention_statistics_folder : string
        Folder where attention statistic information will be stored


    Methods
    -------
    get_excluded_patients()
        Return excluded_patients
    get_data_folders()
        Return data_folders
    get_json_tiling_folder()
        Return json_tiling_folder
    get_attention_folder()
        Return attention_folder
    get_csv_patients()
        Return csv_patients
    get_image_properties()
        Return image_properties
    get_tile_properties()
        Return tile_properties
    get_skip_existing()
        Return skip_existing
    get_split()
        Return split
    get_folds()
        Return folds
    get_monte_carlo()
        Return monte_carlo
    get_monte_carlo_folder()
        Return monte_carlo_folder
    get_narrow_validation()
        Return narrow_validation
    get_feature_folder()
        Return feature_folder
    get_results_folder()
        Return results_folder
    get_attention_statistics_folder()
        Return attention_statistics_folder
    """
    def __init__(self, data_directories, csv_file, working_directory):
        """ 
        Parameters
        ----------
        data_directories : list [string]
            Paths to data directories with WSI
        csv_file : string
            Csv storing patients and their attributes
        working_directory : string
            working directory for data
        """
                # Choose which WSIs to use - detailed explanation in class
        self.image_properties = [
            ImageProperty('HE', 'Hamamatsu', 40, 'FFPE', 20, True)
        ]

        # Choose which Tiles to generate - detailed explanation in class 
        self.tile_properties = [
            TileProperty((256, 256), (0, 0), (256, 256), 
            Compose([RandomCrop(256, 256, always_apply=True, p=1.0), Normalize(always_apply=True, p=1.0)]),
            False, working_directory+' label_maps/',
            True, True, 0.75)
        ]

        # Identifier of patients to manually exclude
        self.excluded_patients = []
        # Data folder - Folder where WSI Images are
        self.data_folders = data_directories
        # Csv of patients with attributes
        self.csv_patients = csv_file
        # JSON Tiling folder
        self.json_tiling_folder = working_directory+'Tiling/'   #TODO
        helper.create_folder(self.json_tiling_folder)
        # Attention Map folder
        self.attention_folder = working_directory+'/Attention/' #TODO
        helper.create_folder(self.attention_folder)


        # Set True if want to Skip existing Tilings - Rerun necessary if changes in tiling/filtering were made
        self.skip_existing = True   #TODO automatic check if Tiling already done --> Tiling needs Identifier for WSI and Tiling Property

        # Split to use as percent 0..1 (Train, Validation, Test)
        self.split = (0.7, 0.1, 0.2)
        # Folds to run --> Not necessary if want to run Monte-Carlo
        self.folds = [0, 1, 2, 3, 4]
        # Set True if want to run Monte Carlo Folds instead of k-Fold
        self.monte_carlo = True
        # Folder to memorize Monte Carlo splits
        self.monte_carlo_folder = working_directory+'Splits/' #TODO
        if self.monte_carlo:
            helper.create_folder(self.monte_carlo_folder)

        self.narrow_validation = False
        # Feature folder
        self.feature_folder = working_directory+'Features/' #TODO
        helper.create_folder(self.feature_folder)


        self.results_folder = working_directory+'Evaluation/Results/'   #TODO
        helper.create_folder(self.results_folder)

        self.attention_statistics_folder = working_directory+'Evaluation/Attention_Statistics/' #TODO
        helper.create_folder(self.attention_statistics_folder)

    def get_excluded_patients(self):
        """ Return excluded_patients
        Returns
        -------
        list [string]
            excluded_patients
        """
        return self.excluded_patients

    def get_data_folders(self):
        """ Return data_folders
        Returns
        -------
        list [string]
            data_folders
        """
        return self.data_folders

    def get_json_tiling_folder(self):
        """ Return json_tiling_folder
        Returns
        -------
        string
            json_tiling_folder
        """
        return self.json_tiling_folder

    def get_attention_folder(self):
        """ Return attention_folder
        Returns
        -------
        string
            attention_folder
        """
        return self.attention_folder

    def get_csv_patients(self):
        """ Return csv_patients
        Returns
        -------
        string
            csv_patients
        """
        return self.csv_patients

    def get_image_properties(self):
        """ Return image_properties
        Returns
        -------
        list [ImageProperty]
            image_properties
        """
        return self.image_properties

    def get_tile_properties(self):
        """ Return tile_properties
        Returns
        -------
        list [TileProperty]
            tile_properties
        """
        return self.tile_properties

    def get_skip_existing(self):
        """ Return skip_existing
        Returns
        -------
        bool
            skip_existing
        """
        return self.skip_existing

    def get_split(self):
        """ Return split
        Returns
        -------
        tuple (float, float, float)
            split
        """
        return self.split

    def get_folds(self):
        """ Return folds
        Returns
        -------
        list [int]
            folds
        """
        return self.folds
        
    def get_monte_carlo(self):
        """ Return monte_carlo
        Returns
        -------
        bool
            monte_carlo 
        """
        return self.monte_carlo

    def get_monte_carlo_folder(self):
        """ Return monte_carlo_folder
        Returns
        -------
        string
            monte_carlo_folder
        """
        return self.monte_carlo_folder

    def get_narrow_validation(self):
        """ Return narrow_valdiation
        Returns
        -------
        bool
            narrow_validation
        """
        return self.narrow_validation

    def get_feature_folder(self):
        """ Return feature_folder
        Returns
        -------
        string
            feature_folder
        """
        return self.feature_folder

    def get_results_folder(self):
        """ Return results_folder
        Returns
        -------
        string
            results_folder
        """
        return self.results_folder

    def get_attention_statistics_folder(self):
        """ Return attention_statistics_folder
        Returns
        -------
        string
            attention_statistics_folder
        """
        return self.attention_statistics_folder