import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv_data(csv_path):
    return pd.read_csv(csv_path)

def guess_server_counts(df):
    return len(df['Server'].unique())

def preprocess_data(df):
    # Handle missing values by replacing them with NaN
    df.replace(['', None], np.nan, inplace=True)
    # Convert columns to numeric (in case they're read as strings)
    numeric_cols = df.columns.drop(['Epoch', 'Server'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df

def filter_epochs(df, begin_epoch, end_epoch):
    return df[(df['Epoch'] >= begin_epoch) & (df['Epoch'] <= end_epoch)]

def calculate_discounted_rewards_per_server(rewards_per_server, discount_factor, episode_duration):
    discounted_rewards_per_server = []

    for server_rewards in rewards_per_server:
        server_discounted_rewards = []
        running_add = 0

        for t in range(len(server_rewards) - 1, -1, -1):
            running_add = running_add * discount_factor + server_rewards.iloc[t]
            server_discounted_rewards.insert(0, running_add)

            if t % episode_duration == 0:
                running_add = 0

        discounted_rewards_per_server.append(server_discounted_rewards)

    return discounted_rewards_per_server

def calculated_summed_rewards_per_server(rewards_per_server, episode_duration):
    summed_rewards_per_server = []

    for server_rewards in rewards_per_server:
        server_summed_rewards = []
        running_sum = 0

        for t in range(len(server_rewards)):
            running_sum += server_rewards.iloc[t]
            server_summed_rewards.append(running_sum)

            if t % episode_duration == 0:
                running_sum = 0

        summed_rewards_per_server.append(server_summed_rewards)

    return summed_rewards_per_server

def plot_data(df, server_counts, episode_duration, discount_factor, batch_size=None):
    _, axs = plt.subplots(2, 2, figsize=(12, 8))

    rewards_per_server = [df[df['Server'] == server]['Average_Reward'] for server in range(server_counts)]
    train_rewards_per_server = [df[df['Server'] == server]['Average_Train_Reward'] for server in range(server_counts)]

    discounted_rewards_per_server = calculate_discounted_rewards_per_server(rewards_per_server, discount_factor, episode_duration)
    discounted_train_rewards_per_server = calculate_discounted_rewards_per_server(train_rewards_per_server, discount_factor, episode_duration)

    summed_rewards_per_server = calculated_summed_rewards_per_server(rewards_per_server, episode_duration)
    summed_train_rewards_per_server = calculated_summed_rewards_per_server(train_rewards_per_server, episode_duration)

    # Assign calculated rewards to their respective servers and epochs
    for server in range(server_counts):
        df.loc[df['Server'] == server, 'Discounted_Reward'] = discounted_rewards_per_server[server]
        df.loc[df['Server'] == server, 'Discounted_Train_Reward'] = discounted_train_rewards_per_server[server]
        df.loc[df['Server'] == server, 'Summed_Reward'] = summed_rewards_per_server[server]
        df.loc[df['Server'] == server, 'Summed_Train_Reward'] = summed_train_rewards_per_server[server]

    if batch_size:
        # Plotting over trained frames
        df['Processed_Frames'] = server_counts * batch_size * df['Epoch']
        df['Processed_Frames_Per_Day'] = server_counts * batch_size * df['Calls_per_sec'] * 60 * 60 * 24

        avg_q_values = df.groupby('Processed_Frames')['Average_Q_Value'].mean()
        avg_train_q_values = df.groupby('Processed_Frames')['Average_Train_Q_Value'].mean()
        axs[0, 0].plot(avg_q_values, label='Average Q Value')
        axs[0, 0].plot(avg_train_q_values, label='Average Train Q Value')
        axs[0, 0].set_xlabel('Processed Frames')
        axs[0, 0].set_ylabel('Values')
        axs[0, 0].set_title('Average Q Values')
        axs[0, 0].legend()

        avg_return = df.groupby('Processed_Frames')['Discounted_Reward'].mean()
        avg_train_return = df.groupby('Processed_Frames')['Discounted_Train_Reward'].mean()
        avg_sum_reward = df.groupby('Processed_Frames')['Summed_Reward'].mean()
        avg_sum_train_reward = df.groupby('Processed_Frames')['Summed_Train_Reward'].mean()
        axs[0, 1].plot(avg_return, label='Average Return')
        axs[0, 1].plot(avg_train_return, label='Average Train Return')
        axs[0, 1].plot(avg_sum_reward, label='Summed Rewards', linestyle='--')
        axs[0, 1].plot(avg_sum_train_reward, label='Summed Train Rewards', linestyle='--')
        axs[0, 1].set_xlabel('Processed Frames')
        axs[0, 1].set_ylabel('Values')
        axs[0, 1].set_title('Average Return and sum of rewards')
        axs[0, 1].legend()

        avg_loss = df.groupby('Processed_Frames')['Loss'].mean()
        axs[1, 0].plot(avg_loss)
        axs[1, 0].set_xlabel('Processed Frames')
        axs[1, 0].set_ylabel('Average Loss')
        axs[1, 0].set_title('Average Loss')

        avg_processed_frames_per_day = df.groupby('Processed_Frames')['Processed_Frames_Per_Day'].mean()
        axs[1, 1].plot(avg_processed_frames_per_day)
        axs[1, 1].set_xlabel('Processed_Frames')
        axs[1, 1].set_ylabel('Avg Processed Frames/Day')
        axs[1, 1].set_title('Avg Processed Frames/Day')
        
    else:
        # Plotting over epochs
        avg_q_values = df.groupby('Epoch')['Average_Q_Value'].mean()
        avg_train_q_values = df.groupby('Epoch')['Average_Train_Q_Value'].mean()
        axs[0, 0].plot(avg_q_values, label='Average Q Value')
        axs[0, 0].plot(avg_train_q_values, label='Average Train Q Value')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Values')
        axs[0, 0].set_title('Average Q Values')
        axs[0, 0].legend()

        avg_return = df.groupby('Epoch')['Discounted_Reward'].mean()
        avg_train_return = df.groupby('Epoch')['Discounted_Train_Reward'].mean()
        avg_sum_reward = df.groupby('Epoch')['Summed_Reward'].mean()
        avg_sum_train_reward = df.groupby('Epoch')['Summed_Train_Reward'].mean()
        axs[0, 1].plot(avg_return, label='Average Return')
        axs[0, 1].plot(avg_train_return, label='Average Train Return')
        axs[0, 1].plot(avg_sum_reward, label='Summed Rewards', linestyle='--')
        axs[0, 1].plot(avg_sum_train_reward, label='Summed Train Rewards', linestyle='--')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Values')
        axs[0, 1].set_title('Average Return')
        axs[0, 1].legend()

        avg_loss = df.groupby('Epoch')['Loss'].mean()
        axs[1, 0].plot(avg_loss)
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Average Loss')
        axs[1, 0].set_title('Average Loss')

        avg_calls_per_sec = df.groupby('Epoch')['Calls_per_sec'].mean()
        axs[1, 1].plot(avg_calls_per_sec)
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Average Calls per Second')
        axs[1, 1].set_title('Average Calls per Second')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv_path", type=str, help="Path to CSV file")
    parser.add_argument("-b", "--begin_epoch", type=int, default=0, help="Beginning epoch")
    parser.add_argument("-e", "--end_epoch", type=int, default=None, help="End epoch")
    parser.add_argument("-d", "--episode_duration", type=int, default=1, help="Episode duration in epochs")
    parser.add_argument("-g", "--discount_factor", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("-s", "--batch_size", type=int, default=None, help="Batch size")

    args = parser.parse_args()

    df = read_csv_data(args.csv_path)
    server_counts = guess_server_counts(df)
    df = preprocess_data(df)

    if args.end_epoch is None:
        args.end_epoch = df['Epoch'].max()

    filtered_df = filter_epochs(df, args.begin_epoch, args.end_epoch)
    plot_data(filtered_df, server_counts, args.episode_duration, args.discount_factor, args.batch_size)
