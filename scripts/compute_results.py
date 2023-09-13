import argparse
import pandas as pd
import numpy as np
from glob import glob
from termcolor import cprint


def calculate_average_return(df):
    """
    Calculate the average return for each checkpoint in the DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    
    Returns:
    Series: A Pandas Series containing the average return for each checkpoint.
    """
    # Extract columns containing 'return' in their names
    return_columns = [col for col in df.columns if 'return' in col]
    
    # Calculate the average return for each checkpoint
    avg_return = df[return_columns].mean(axis=1)
    
    return avg_return


def main(file_paths, method):
    """
    Main function to execute the tasks.
    
    Parameters:
    file_paths (list): List of file paths for the seeds.
    method (str): Method to select the checkpoint ('best' or 'last').
    """
    # List to store average return for selected checkpoint across seeds
    selected_avg_return = []
    
    # Loop through each file and calculate the average return
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        avg_return = calculate_average_return(df)
        
        # Select the checkpoint based on the method
        if method == 'best':
            selected_return = avg_return.max()
            # get the index of the best checkpoint
            best_checkpoint = avg_return.idxmax()
            print(f"Best checkpoint for {file_path}: {best_checkpoint}, with average return: {selected_return:.2f}")
        elif method == 'last':
            selected_return = avg_return.iloc[-1]
        else:
            print(f"Unknown method: {method}. Skipping this seed.")
            continue
        
        selected_avg_return.append(selected_return)
    
    # Calculate the average and standard deviation over all seeds
    avg_over_seeds = np.mean(selected_avg_return)
    std_over_seeds = np.std(selected_avg_return)
    
    cprint(f"Average return over all seeds: {avg_over_seeds:.2f}", 'cyan')
    cprint(f"Standard deviation over all seeds: {std_over_seeds:.2f}", 'cyan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average return across seeds and tasks.')
    parser.add_argument('--file_paths', nargs='+', required=True, help='List of file paths for the seeds.')
    parser.add_argument('--method', choices=['best', 'last'], help='Method to select the checkpoint ("best" or "last").', default='last')
    
    args = parser.parse_args()
    main(args.file_paths, args.method)
