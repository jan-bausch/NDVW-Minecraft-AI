import matplotlib.pyplot as plt
import pandas as pd
import sys

action_names = {
    0: 'Jumping',
    1: 'Going Left',
    2: 'Going Right',
    3: 'Going Forward',
    4: 'Going Backward',
    5: 'Placing Block',
    6: 'Breaking Block',
    7: 'Looking Left',
    8: 'Looking Right',
    9: 'Looking Up',
    10: 'Looking Down'
}

if len(sys.argv) != 2:
    print("Usage: python script_name.py data.csv")
    sys.exit(1)

# Read data from the CSV file
file_name = sys.argv[1]
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print("File not found.")
    sys.exit(1)

# Convert string representation of lists to actual lists
df['q_values'] = df['q_values'].apply(lambda x: list(map(float, x.split())))
df['q_target_values'] = df['q_target_values'].apply(lambda x: list(map(float, x.split())))

# Calculate mean of every 10 rows for q_values and q_target_values
window_size = 50
mean_q_values = df['q_values'].apply(pd.Series).rolling(window_size, min_periods=1).mean()
mean_q_target_values = df['q_target_values'].apply(pd.Series).rolling(window_size, min_periods=1).mean()

# Calculate mean of every 10 rows for trained_frames_count
mean_trained_frames_count = df['trained_frames_count'].rolling(window_size, min_periods=1).mean()

num_cols = len(mean_q_target_values.columns)
color_map = plt.get_cmap('Set3')

# Plotting mean_q_values
for i in range(len(mean_q_values.columns)):
    color = color_map(i / num_cols)
    plt.plot(
        mean_trained_frames_count,
        mean_q_values[i],
        marker='o',
        linestyle='-',
        markersize=1,
        label=f'{action_names[i]}',
        color=color,
    )

# # Plotting mean_q_target_values
# for i in range(len(mean_q_target_values.columns)):
#     color_index = i % 11
#     plt.plot(
#         mean_trained_frames_count,
#         mean_q_target_values[i],
#         marker='x',
#         linestyle='-',
#         markersize=1,
#         label=f'target {action_names[i]}',
#         color=f'C{color_index}',
#     )

plt.xlabel('Trained Frames Count')
plt.ylabel('Mean Q Values')
plt.title(f'Mean Q Values over Trained Frames Count (Window Size = {window_size})')
plt.legend()
plt.grid(True)
plt.show()
