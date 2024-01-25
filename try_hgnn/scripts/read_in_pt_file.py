import torch

# Load the file
PATH="/home/almas/projects/def-gregorys/almas/OpenHGNN/openhgnn/output/fastGTN/fastGTN_imdb4GTN_node_classification.pt"

model = torch.load(PATH) 
print(model)
