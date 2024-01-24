import torch

# Load the file
pt_file = torch.load("/home/almas/projects/def-gregorys/almas/OpenHGNN/openhgnn/output/fastGTN/fastGTN_imdb4GTN_node_classification.pt")

# Print the head of the file
print(pt_file[:5])