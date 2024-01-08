import pandas as pd
import sys

def remove_duplicates_except_last(csv_path):
    # Read the CSV file into a pandas DataFrame
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return
    
    # Remove duplicates except for the last occurrence based on 'episode', 'step', and 'world_id' columns
    cleaned_data = data.drop_duplicates(subset=['episode', 'step', 'world_id'], keep='last')
    
    # Save the cleaned data to a new CSV file
    cleaned_data.to_csv('cleaned_' + csv_path, index=False)
    print(f"Duplicates removed except for the last occurrence. Cleaned data saved to cleaned_{csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
    else:
        csv_path = sys.argv[1]
        remove_duplicates_except_last(csv_path)
