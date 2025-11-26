# shared/model_utils.py
import torch
import os
from collections import OrderedDict

def save_model_torch(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model_torch(model, path, map_location=None):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model

def get_parameters_numpy(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters_from_numpy(model, parameters):
    state_dict = model.state_dict()
    new_state = OrderedDict()
    for (k, _), arr in zip(state_dict.items(), parameters):
        new_state[k] = torch.tensor(arr)
    model.load_state_dict(new_state)

def set_state_dict_from_numpy_by_keys(model, weights_dict):
    """
    Update a PyTorch model's state_dict using only specific keys 
    provided as numpy arrays.
    """
    state_dict = model.state_dict()

    for key, value in weights_dict.items():
        if key in state_dict:
            state_dict[key] = torch.tensor(value)
        else:
            print(f"Warning: key '{key}' not found in model state dict")

    model.load_state_dict(state_dict)
    return model
