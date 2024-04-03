import csv
import pandas as pd 
import numpy as np 
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os  # Import the os module

def read_data(input_file): 
    return pd.read_csv(input_file)  # Return the DataFrame

def line_loss_alt(data, X, Y, output_folder, output_file):
    # Append type information to X and Y if needed, e.g., ":Q" for quantitative data
    chart = alt.Chart(data).mark_line().encode(
        x=X + ":Q",  # Assuming X is quantitative. Change ":Q" as needed
        y=Y + ":Q"  # Assuming Y is quantitative. Change ":Q" as needed
    )
    chart.save(os.path.join(output_folder, output_file + ".html"))  # Use os.path.join for path

def line_loss_sns(data, X, Y, output_folder, output_file):
    sns_plot = sns.lineplot(data=data, x=X, y=Y)
    plt.savefig(os.path.join(output_folder, output_file + ".svg"))  # Use os.path.join for path
    plt.savefig(os.path.join(output_folder, output_file + ".png"))  # Use os.path.join for path
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse log file and write to CSV')
    parser.add_argument('input_file', help='Input log file path')
    parser.add_argument('output_folder', help='Output folder path')
    parser.add_argument('output_file', help='Output file path and name')
    parser.add_argument('X',help="X axis")
    parser.add_argument('Y',help="Y axis")
    args = parser.parse_args()

    data = read_data(args.input_file)  # Make sure to assign the returned DataFrame to data

    # Ensure output_folder exists or create it
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    line_loss_alt(data=data, X=args.X, Y=args.Y, output_folder=args.output_folder, output_file=args.output_file)
    line_loss_sns(data=data, X=args.X, Y=args.Y, output_folder=args.output_folder, output_file=args.output_file)

# data= read_data("/project/def-gregorys/almas/OpenHGNN/try_hgnn/data/2024-Mar-28_1049_fastGTN_node_classification_imdb4GTN.csv")
# line_loss_alt(data=data, X="Epoch", Y="Valid Loss", output_folder="/project/def-gregorys/almas/OpenHGNN/try_hgnn/img/", output_file="valid_loss_curve")
# line_loss_alt(data=data, X="Epoch", Y="Train Loss", output_folder="/project/def-gregorys/almas/OpenHGNN/try_hgnn/img/", output_file="train_loss_curve")
# line_loss_sns(data=data, X="Epoch", Y="Valid Loss", output_folder="/project/def-gregorys/almas/OpenHGNN/try_hgnn/img/", output_file="valid_loss_curve")
# line_loss_sns(data=data, X="Epoch", Y="Train Loss", output_folder="/project/def-gregorys/almas/OpenHGNN/try_hgnn/img/", output_file="train_loss_curve")
