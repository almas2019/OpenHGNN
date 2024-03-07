import torch
import altair as alt
import pandas as pd
output_folder="/home/almas/projects/def-gregorys/almas/OpenHGNN/openhgnn/output/fastGTN/"
# Initialize an empty list to store tensors for each layer
layer_tensors = []

# Load tensors from files and compute average for each layer
for x in range(4):
    layer_tensors_x = []
    for y in range(2):
        file_path = output_folder+f"filters_layer_{x}_filter_{y}.pt"
        tensor = torch.load(file_path)
        layer_tensors_x.append(tensor)
    # Compute average across tensors for each layer
    layer_average_x = torch.mean(torch.stack(layer_tensors_x), dim=0)
    layer_tensors.append(layer_average_x)

# Combine layer tensors into a 4x5 matrix
combined_matrix = torch.stack(layer_tensors)

# Extract the labels (MD, DM, MA, AM, I)
labels = ['MD', 'DM', 'MA', 'AM', 'I']

# Create a list to store the data in the correct format
data = []

# Iterate through the tensor data and reshape it
for i, layer in enumerate(combined_matrix):
    for j, value in enumerate(layer):
        data.append({'Layer': i, 'Label': labels[j], 'Value': value.item()})

# Convert the reshaped data to a pandas DataFrame
df = pd.DataFrame(data)


# Create the heatmap using Altair
heatmap = alt.Chart(df).mark_rect().encode(
    x='Layer:O',
    y='Label:O',
    color=alt.Color('Value:Q').scale(scheme="blues") , # blues colour scheme 
    sort=None,
).properties(
    title='Heatmap of Filters',
    width=200,
    height=150
)
# Save the heatmap as an HTML file
heatmap.save(output_folder+'heatmap.html')


df.to_csv(output_folder+"imdb_layers_avg_channel.csv")