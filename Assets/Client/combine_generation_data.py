import pandas as pd
import glob

# Get a list of all CSV files that match the pattern
csv_files = glob.glob('generation_server_*_data.csv')

# Initialize an empty list to hold DataFrames
dfs = []

# Process each CSV file
for csv_file in csv_files:
    # Load the data into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract the server number from the filename
    server_number = int(csv_file.split('_')[2])

    # Add a new 'world_id' column
    df['world_id'] = server_number * df['world_id'].max()

    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
result_df = pd.concat(dfs)

# Save the resulting DataFrame to a new CSV file
result_df.to_csv('combined_data.csv', index=False)