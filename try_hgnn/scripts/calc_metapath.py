import pandas as pd
import numpy as np
import itertools

# Read the CSV data into a pandas DataFrame
data = pd.read_csv("/home/almas/projects/def-gregorys/almas/OpenHGNN/openhgnn/output/fastGTN/imdb_layers_avg_channel.csv")


# Organize the data into a dictionary of dictionaries
#This code organizes the data from the DataFrame into a nested dictionary structure called edge_weights. It iterates over each row in the DataFrame, extracting the layer, label, edge type, and value.
edge_weights = {}
for index, row in data.iterrows():
    layer = row['Layer']
    label = row['Label']
    value = row['Value']
    edge_weights[(layer, label)]= value

# Define a function to calculate metapaths and their weights
def calculate_metapaths(edge_weights):
    metapaths = []
    for length in range(1, 5):  # Generate metapaths of lengths 1 to 4
        for combo in itertools.product(edge_weights.keys(), repeat=length):
            if len(set(combo)) == len(combo):  # Check if all edge types in the metapath are unique
                weight_product = np.prod([edge_weights[edge] for edge in combo])
                metapaths.append((length, combo, weight_product))
    metapaths.sort(key=lambda x: x[2], reverse=True)  # Sort metapaths by weight
    return metapaths

# Calculate top metapaths
top_metapaths = calculate_metapaths(edge_weights)

# Print top metapaths and their weights
for idx, (length, metapath, weight) in enumerate(top_metapaths):
    print(f"Top {idx + 1} Metapath: Length {length}, Metapath {metapath}, Weight: {weight}")
