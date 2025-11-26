# server/model.py
from shared.model import get_model as get_shared_model

# Thin wrapper to keep server/model.py import consistent with previous layout.
def get_model(in_dim=561, num_classes=6):
    return get_shared_model(in_dim=in_dim, num_classes=num_classes)
