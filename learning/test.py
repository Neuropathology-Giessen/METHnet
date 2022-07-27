from xml.dom.minidom import Identified
import torch
from learning.train import collate_MIL as collate_MIL 

from models.attention_model import Attention_MB
from learning.train import MIL_Dataset as MIL_Dataset
from learning.train import calculate_error as calculate_error

from torch.utils.data import DataLoader, SequentialSampler
import numpy as np



def test(test_patients, fold, setting, draw_map=True):
    """ Run model prediction for patients

    Parameters
    ----------
    test_patients : list [[Patient]]
        List containign a list per class containing Patient to predict
    fold : int
        Current fold to use / or run for Monte Carlo
    setting : Setting
        Setting object defined according to class
    draw_map : bool
        True if want to set attention map

    Returns
    -------
    float
        Balanced accuracy
    float
        Sensititivy
    float
        Specificity
    """
    # Get number of classes
    network_setting = setting.get_network_setting()
    n_classes = setting.get_class_setting().get_n_classes()
    
    # Create model
    model = Attention_MB(setting)
    model.relocate()

    # Flatten patient list
    patients_test = []

    for patients in test_patients:
        for patient in patients:
            patients_test.append(patient)
    # Create dataset
    test_dataset = MIL_Dataset(patients_test)
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, sampler = SequentialSampler(test_dataset), collate_fn=collate_MIL)
    # Get model folder
    model_folder = network_setting.get_model_folder()
    model_file = 's_{}_checkpoint.pt'.format(fold)

    # Load model
    model.load_state_dict(torch.load(model_folder+model_file))
    # Predict patients and compute balanced accuracy
    balanced_accuracy, sensitivity, specificity = test_model(model, n_classes, test_loader, patients_test, draw_map)

    return balanced_accuracy, sensitivity, specificity

def test_model(model, n_classes, loader, patients, draw_map):
    """ Predict patients

    Parameters
    ----------
    model : Attention_MB
        The attention model
    n_classes : int
        The number of possible classes
    loader : DataLoader
        The data loader
    patients : list [Patient]
        The list of patients to predict
    draw_map : bool
        True if want to save attention maps

    Returns
    -------
    float
        Balanced accuracy
    float
        Sensititivy
    float
        Specificity
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # Values for sensitivity and specificity
    error_class_wise = np.zeros(n_classes)
    counter_class_wise = np.zeros(n_classes)

    # Predict all patients
    with torch.no_grad():
        for batch_idx, (data, label, keys, identifier) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            # predict patient
            logits, Y_prob, Y_hat, A = model(data, label=label)
            # Calculate error between predicted class and label
            error = calculate_error(Y_hat, label)
            # Update values for sensitivity and specificity
            error_class_wise[label] += error
            counter_class_wise[label] += 1
            # Get correct patient
            patient = [p for p in patients if p.get_identifier() == identifier[0][0]][0]
            # Append score to diagnosis
            patient.get_diagnosis().add_predicted_score(Y_prob.cpu()[0][label.cpu()].numpy()[0])
            # Get attention map for predicted class
            A = A[Y_hat]
            A = A.view(-1, 1).cpu().numpy()
            # Get tile keys in attention map
            keys = keys.cpu().numpy()[0]
            if draw_map:
                # Save attention map
                patient.set_map(A, keys)
  

    # Compute sensitivity and specificity
    sensitivity = (counter_class_wise[0] - error_class_wise[0]) / counter_class_wise[0]
    specificity = (counter_class_wise[1] - error_class_wise[1]) / counter_class_wise[1]
    # Compute balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2.

    return balanced_accuracy, sensitivity, specificity