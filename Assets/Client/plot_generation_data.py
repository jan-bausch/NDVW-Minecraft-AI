import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


# Get the filename from command line arguments
if len(sys.argv) < 2:
    print("Please provide the filename as a command line argument.")
    sys.exit(1)

filename = sys.argv[1]

# Set default values for optional arguments
beginning_episode = 1
end_episode = None
sample_percentage = 100

# Check if optional arguments are provided
if len(sys.argv) >= 3:
    beginning_episode = int(sys.argv[2])
if len(sys.argv) >= 4:
    end_episode = int(sys.argv[3])
if len(sys.argv) >= 5:
    sample_percentage = float(sys.argv[4])

# Load the data
df = pd.read_csv(filename)

# Apply optional arguments
if end_episode is not None:
    df = df[(df['episode'] >= beginning_episode) & (df['episode'] <= end_episode)]
if sample_percentage < 100:
    df = df.sample(frac=sample_percentage/100, random_state=1)

# # sample episodes (but not the steps within episodes) to only take 10% of the data
# df = df.sample(frac=0.1, random_state=1)

# Create a figure with 4 subplots
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1, 2]})

world_ids = df['world_id'].unique()

# # Calculate the correlation matrix
# corr_df = pd.DataFrame()
# corr_df['world_id'] = df['world_id']

# after = 20
# # Correlation between, action was taken and sum of reward 10 steps after
# corr_df['action'] = df['action']
# for world_id in world_ids:
#     corr_df.loc[corr_df['world_id'] == world_id, f'sum_reward_{after}_steps_after'] = df[df['world_id'] == world_id]['reward'].rolling(window=after).sum().shift(-after)
# #corr_df = corr_df.groupby('action').mean()

# # one hot encode action
# one_hot = pd.get_dummies(corr_df['action'])
# corr_df = corr_df.drop('action', axis=1)
# corr_df = corr_df.join(one_hot)

# corr_df = corr_df.corr().iloc[0, 2:]

# print(corr_df)

# # bar plot of correlation between action taken and sum of reward 10 steps after
# corr_df.plot(kind='bar', ax=axs[0, 0], color='blue', alpha=0.7)
# ax1.set_title(f'Bar Plot of Correlation between Action and Sum of Reward {after} Steps After')
# ax1.set_xlabel('Action')
# ax1.set_ylabel('Correlation')


# Plot 2: Bar plot of actions
action_frequencies = df['action'].value_counts(normalize=True)
action_frequencies.plot(kind='bar', ax=ax1, color='green', alpha=0.7)
ax1.set_title('Bar Plot of Actions')
ax1.set_xlabel('Action')
ax1.set_ylabel('Frequency')

# Plot: Line plot of reward over steps for all episodes
window = 1
reward_sum_per_episode = df.groupby(['episode', 'world_id'])['reward'].sum()
reward_sum_per_episode = reward_sum_per_episode.groupby('episode').max()
reward_sum_per_episode_avg = reward_sum_per_episode.rolling(window=window).mean()

ax2.plot(reward_sum_per_episode_avg.index, reward_sum_per_episode_avg.values, color='red')
ax2.set_title(f'Line Plot of Reward Sum per Episode')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Reward Sum')


# sample 20 random episodes
episodes_df = pd.DataFrame()
episodes_df['episode'] = df['episode'].unique()
episodes_df = episodes_df.sample(frac=1.0, random_state=1)
episodes_df = episodes_df.sort_values(by='episode')
new_df = episodes_df.merge(df, on='episode')

# steps_df = pd.DataFrame()
# steps_df['step'] = df['step'].unique()
# steps_df = steps_df[steps_df['step'] % 10 == 0]
# steps_df = steps_df.sort_values(by='step')
# new_df = new_df.merge(steps_df, on='step')

# change episodes to have numbers from 0 to 19 depending on the current rank of their value
new_df['episode'] = new_df['episode'].rank(method='dense').astype(int) - 1
new_df['step'] = new_df['step'].rank(method='dense').astype(int) - 1
grouped = new_df.groupby(['episode', 'step', 'action']).size().reset_index(name='frequency')

print(grouped.head(20))

actions = grouped['action'].unique()  # Get unique action values

# Assign different colors to different actions
distinct_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'olive']

for i, action in enumerate(actions):
    action_data = grouped[grouped['action'] == action]
    ax3.scatter(action_data['episode'] + (action_data['step']/300), action_data['frequency'],
               color=distinct_colors[i], label=f'Action {action}', alpha=1, marker='x', s=10)

ax3.set_xlabel('Time (Episodes + Steps)')
ax3.set_ylabel('Frequency')
ax3.legend()

# Plot: Heatmap of action vs reward
# after = 30

# new_df = pd.DataFrame()
# new_df['action'] = df['action'].repeat(after).reset_index(drop=True)

# reward_df = pd.DataFrame()
# reward_df['world_id'] = df['world_id']
# for n in range(1, after+1):
#     for world_id in world_ids:
#         reward_df.loc[reward_df['world_id'] == world_id, f'reward_{n}'] = df[df['world_id'] == world_id]['reward'].shift(-n)

# for n in range(1, after+1):
#     new_df.loc[new_df.index % n == 0, 'reward'] = reward_df[f'reward_{n}']
#     new_df.loc[new_df.index % n == 0, 'n'] = str(n)

# new_df = new_df.groupby(['action', 'n']).mean().reset_index()
# new_df['reward'] = new_df['reward'].fillna(0)

# matrix_df = pd.DataFrame()

# for i in range(1, after+1):
#     row_df = pd.DataFrame()
#     row_df['reward'] = new_df[new_df['n'] == str(i)]['reward']
    
#     row_df['action'] = new_df['action']
#     #one hot encode actions
#     one_hot = pd.get_dummies(row_df['action'])
#     row_df = row_df.drop('action', axis=1)
#     row_df = row_df.join(one_hot)

#     row_df = row_df.corr().iloc[0, 1:]

#     # put the row_df into the matrix_df
#     matrix_df[i] = row_df

# heatmap = axs[1, 1].imshow(matrix_df.T, cmap='hot', aspect='auto')
# cbar = plt.colorbar(heatmap, ax=axs[1, 1])
# axs[1, 1].set_title('Heatmap of Action correlation w/ Reward N step after')
# axs[1, 1].set_xlabel('Action')
# axs[1, 1].set_ylabel('N step after')
# axs[1, 1].set_xticks(range(0, 11))
# axs[1, 1].set_yticks(range(0, after+1))

# Show the plots
plt.tight_layout()
plt.show(block=True)
