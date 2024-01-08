import pandas as pd
import sys

def find_duplicates(csv_path):
    # Read the CSV file into a pandas DataFrame
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return
    
    # Identify duplicates based on 'episode', 'step', and 'world_id' columns
    duplicate_rows = data[data.duplicated(subset=['episode', 'step', 'world_id'], keep=False)]
    
    if duplicate_rows.empty:
        print("No duplicates found for specified columns.")
    else:
        print("Duplicate rows based on 'episode', 'step', and 'world_id':")
        print(duplicate_rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
    else:
        csv_path = sys.argv[1]
        find_duplicates(csv_path)
