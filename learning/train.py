import torch

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import torch.optim as optim
from models.attention_model import Attention_MB

from learning.logger import EarlyStopping, Accuracy_Logger
import os
import numpy as np

class MIL_Dataset(Dataset):
    """ Dataset for DataLoader

    Attributes
    ----------
    X : list [Patient]
        List of patient used to train network

    Methods
    -------
    __len__()
        Number of patients
    __getitem__(idx)
        Get features, labels, keys and identifier for current patient
    """
    def __init__(self, X):
        """
        Parameters
        ----------
        X : list [Patient]
            List of patients used to train network
        """
        self.X = X

    def __len__(self):
        """ Get number of batches to process

        Returns 
        -------
        int
            Number of patients/batches to process
        """
        return len(self.X)

    def __getitem__(self, idx):
        """ Get features, label, keys and identifier for current patient

        Parameters
        ----------
        idx : int
            Index of patient in list X
        
        Returns
        -------
        TorchTensor
            Encoded features for this patient
        int
            class label of patient
        list [(int, int)] 
            keys recognizing Tiles same order as features
        string
            patient identifier
        """
        # One idx refers to one patients
        patient = self.X[idx]
        features, keys = patient.get_features()
        features = features.astype(np.float32)
        features = torch.from_numpy(features)
        
        label = patient.get_diagnosis().get_label()
        identifier = patient.get_identifier()

        return features, label, keys, identifier


def collate_MIL(batch):
    """ Cast batch to Tensors
    Parameters
    ----------
    batch : list [(TorchTensor, int, list[(int, int)], string)]
        Patients in batch
    Returns
    -------
    list 
        List of Tensors to use for training containing the patient information
    """
    # Concatenate images
    img = torch.cat([item[0] for item in batch], dim=0)
    # Concatenate labels
    label = torch.LongTensor([item[1] for item in batch])
    # Concatenate keys
    keys = torch.LongTensor([item[2] for item in batch])
    # Concatenate identifiers
    identifier = [[item[3] for item in batch]]

    return [img, label, keys, identifier]

def calculate_error(Y_hat, Y):
    """ Calculate error of classification
    Parameters
    ----------
    Y_hat : Tensor
        Predicted class
    Y : Tensor
        True class
    """
    error = 1. -Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def train(train_patients, validation_patients, fold, setting):
    """ Train an Attention model and save the best performing one

    Parameters
    ----------
    train_patients : list [[Patient]]
        List of list per class of patients to train the model on
    validation_patients : list [[Patient]]
        List of list per class of patients to validate the model on
    fold : int
        Current fold used or iteration of Monte-Carlo cross validation
    setting : Setting
        Setting as defined by class
    """
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Number of classes
    n_classes = setting.get_class_setting().get_n_classes()
    # Create Attention model
    model = Attention_MB(setting)
    model.relocate()
    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
    # Early stopping checker
    early_stopping = EarlyStopping(patience=setting.get_network_setting().get_patience(), stop_epoch=setting.get_network_setting().get_stop_epoch(), verbose=setting.get_network_setting().get_verbose())
    # Folder to save model parameters to
    model_folder = setting.get_network_setting().get_model_folder()
    # File to save model parameters to
    model_file = 's_{}_checkpoint.pt'.format(fold)
    if os.path.exists(model_folder + model_file):
        return
    # Iterate epochs
    for epoch in range(setting.get_network_setting().get_epochs()):
        # Get minimum number of patients for one class in train set
        min_n = min([len(train_patients[i]) for i in range(len(train_patients))])

        patients_train = []
        # Iterate over classes
        for patients in train_patients:
            # Append minimum number of patients for this class to current epoch dataset
            indices = np.arange(0, len(patients))
            np.random.shuffle(indices)

            for i in range(min_n):
                patients_train.append(patients[indices[i]])

        # Get minimum number of patients for one class in validation set
        min_n = min([len(validation_patients[i]) for i in range(len(validation_patients))])

        patients_validation = []
        # Iterate over classes
        for patients in validation_patients:
            # Append minimum number of patients for this class to current epoch dataset
            indices = np.arange(0, len(patients))
            np.random.shuffle(indices)

            for i in range(min_n):
                patients_validation.append(patients[indices[i]])
        # Create datasets and Loaders
        train_dataset = MIL_Dataset(patients_train)
        train_loader = DataLoader(train_dataset, batch_size=1, sampler=RandomSampler(train_dataset), collate_fn=collate_MIL)

        validation_dataset = MIL_Dataset(patients_validation)
        validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=SequentialSampler(validation_dataset), collate_fn=collate_MIL)
        # Train one epoch
        train_epoch(epoch, model, n_classes, train_loader, loss_fn, optimizer)
        # Validate the model
        stop = validate_epoch(epoch, model, n_classes, validation_loader, loss_fn, early_stopping, ckpt_name=model_folder+model_file)
        # Apply Early stopping
        if stop and setting.get_network_setting().get_early_stopping():
            break



def train_epoch(epoch, model, n_classes, loader, loss_fn, optimizer):
    """ One epoch of model training

    Parameters
    ----------
    epoch : int
        Current epoch
    model : AttentionMB
        Model to train
    n_classes : int
        Number of classes in classification problem
    loader : DataLoader
        DataLoader holding the train data
    loss_fn : torch.nn.CrossEntropyLoss
        Loss function
    optimizer : Adam
        Optimizer to train model with
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    acc_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    # Iterate batches
    for batch_idx, (data, label, keys, identifier) in enumerate(loader):
        
        data, label = data.to(device), label.to(device)
        # Forward pass
        logits, Y_prob, Y_hat, A = model(data, label=label)
        
        # Compute loss and log performance
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        total_loss = loss

        train_loss += loss_value

        error = calculate_error(Y_hat, label)
        train_error += error
        # Backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))


def validate_epoch(epoch, model, n_classes, loader, loss_fn, early_stopping, ckpt_name):
    """ Validate model and save if new best score is achieved
    Parameters
    ----------
    epoch : int
        Current epoch
    model : AttentionMB
        Model to validate
    n_classes : int
        Number of classes in classification problem
    loader : DataLoader
        DataLoader holding validation data
    loss_fn : torch.nn.CrossEntropyLoss
        Loss function
    early_stopping : EarlyStopping
        EarlyStopping as defined by class
    ckpt_name : string
        File to save model parameters to

    Returns
    -------
    bool
        True if early stopping should be applied
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)

    val_loss = 0.
    val_error = 0.
    # Just forward pass needed
    with torch.no_grad():
        for batch_idx, (data, label, keys, identifier) in enumerate(loader):
            
            data, label = data.to(device), label.to(device)
            # Forward pass
            logits, Y_prob, Y_hat, A = model(data, label=label)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            error = calculate_error(Y_hat, label)

            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)
    # Compute Early stopping
    early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)

    if early_stopping.early_stop:
        return True

    return False