import torch
from openhgnn.models import fastGTN
# Load the file
PATH="/home/almas/projects/def-gregorys/almas/OpenHGNN/openhgnn/output/fastGTN/fastGTN_imdb4GTN_node_classification.pt"

import torch

# Define the configuration for IMDb dataset
config_imdb = {
    'lr': 0.01,
    'num_layers': 3,
    'dropout': 0.3,
    'num_channels': 4,
    'hidden_dim': 128,
    'adaptive_lr_flag': False
}

# Instantiate the fastGTN model with the provided configuration
model = fastGTN(**config_imdb)

# Load the pretrained model weights from the .pt file
model_state_dict = torch.load(PATH)
model.load_state_dict(model_state_dict['model_state_dict'])

# Set the model to evaluation mode
model.eval()