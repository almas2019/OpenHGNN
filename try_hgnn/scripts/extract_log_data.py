import re
import csv
import argparse

def parse_log(log_file):
    """Parse log file
     """
    # Define the pattern to extract relevant information for filename
    filename_pattern = r'(\w{3} \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2}).*Model: (\w+),\s*Task: (\w+),\s*Dataset: (\w+)'

    # Define the regex patterns for data extraction
    regex_epoch = re.compile(r'Epoch: (\d+)')
    regex_train_loss = re.compile(r'Train loss: ([\d.]+)')
    regex_valid_loss = re.compile(r'Valid loss: ([\d.]+)')
    regex_f1_scores = re.compile(r'Mode:(\w+), Macro_f1: ([\d.]+); Micro_f1: ([\d.]+);')

    # Initialize lists to store extracted data
    epochs = []
    train_losses = []
    valid_losses = []
    train_macro_f1s = []
    train_micro_f1s = []
    valid_macro_f1s = []
    valid_micro_f1s = []

    # Read the log file
    with open(log_file, 'r') as file:
        log_data = file.readlines()

    # Extract information for filename
    match_filename = re.search(filename_pattern, log_data[0])
    if match_filename:
        datetime_str = match_filename.group(1)
        model = match_filename.group(2)
        task = match_filename.group(3)
        dataset = match_filename.group(4)
        filename_info = f"{datetime_str.replace(' ', '_')}_{model}_{task}_{dataset}.csv"
    else:
        print("Could not extract filename information from log file.")
        exit()

    # Extract data
    for line in log_data:
        epoch_match = regex_epoch.search(line)
        if epoch_match:
            epoch = epoch_match.group(1)
            train_loss = regex_train_loss.search(line).group(1)
            valid_loss = regex_valid_loss.search(line).group(1)

            f1_scores_text = line
            f1_matches = regex_f1_scores.findall(f1_scores_text)
            if f1_matches:
                for match in f1_matches:
                    mode, macro_f1, micro_f1 = match
                    if mode == 'train':
                        train_macro_f1s.append(macro_f1)
                        train_micro_f1s.append(micro_f1)
                    elif mode == 'valid':
                        valid_macro_f1s.append(macro_f1)
                        valid_micro_f1s.append(micro_f1)

            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

    return filename_info, epochs, train_losses, valid_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s


def write_csv(output_folder, filename_info, epochs, train_losses, valid_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s):
    if epochs:
        output_file = os.path.join(output_folder, filename_info)
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train Loss', 'Valid Loss', 'Train Macro F1', 'Train Micro F1', 'Valid Macro F1', 'Valid Micro F1'])
            writer.writerows(zip(epochs, train_losses, valid_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s))
        print(f"Data has been extracted and saved to {output_file}")
    else:
        print("No data found in the log file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse log file and write to CSV')
    parser.add_argument('input_file', help='Input log file path')
    parser.add_argument('output_folder', help='Output folder path')
    args = parser.parse_args()

    filename_info, epochs, train_losses, valid_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s = parse_log(args.input_file)
    write_csv(args.output_folder, filename_info, epochs, train_losses, valid_losses, train_macro_f1s, train_micro_f1s, valid_macro_f1s, valid_micro_f1s)