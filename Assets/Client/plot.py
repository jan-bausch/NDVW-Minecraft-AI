import argparse
import csv
import matplotlib.pyplot as plt

S = 10

def plot_multiple_averages(csv_filename, ignore_entries=0):
    epochs = []
    avg_q_values = []
    train_avg_q_values = []
    avg_rewards = []
    train_avg_rewards = []
    avg_losses = []
    calls_per_sec = []

    with open(csv_filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(csv_reader):
            if i < ignore_entries:
                print(f'Ignoring row {i}')
                continue  # Skip the initial rows

            epoch = int(row['Epoch'])
            epochs.append(epoch)

            if 'Average_Q_Value' in row and row['Average_Q_Value'] != '':
                avg_q_values.append(float(row['Average_Q_Value']))
            else:
                avg_q_values.append(None)

            if 'Average_Train_Q_Value' in row and row['Average_Train_Q_Value'] != '':
                train_avg_q_values.append(float(row['Average_Train_Q_Value']))
            else:
                train_avg_q_values.append(None)

            if 'Average_Reward' in row and row['Average_Reward'] != '':
                avg_rewards.append(float(row['Average_Reward']))
            else:
                avg_rewards.append(None)

            if 'Average_Train_Reward' in row and row['Average_Train_Reward'] != '':
                train_avg_rewards.append(float(row['Average_Train_Reward']))
            else:
                train_avg_rewards.append(None)

            if 'Loss' in row and row['Loss'] != '':
                avg_losses.append(float(row['Loss']))
            else:
                avg_losses.append(None)

            if 'Calls_per_sec' in row and row['Calls_per_sec'] != '':
                calls_value = float(row['Calls_per_sec'])
                calls_per_sec.append(calls_value)
            else:
                calls_per_sec.append(None)

    print(train_avg_rewards)
    print(avg_rewards)

    # Plotting the averages
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].scatter(epochs, avg_q_values, label='Average Q-value', s=S)
    axs[0, 0].scatter(epochs, train_avg_q_values, label='Average Train Q-value', s=S)
    axs[0, 0].set_title('Average Q-value vs Average Train Q-value')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].scatter(epochs, avg_rewards, label='Average Reward', s=S)
    axs[0, 1].scatter(epochs, train_avg_rewards, label='Average Train Reward', s=S)
    axs[0, 1].set_title('Average Reward vs Average Train Reward')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].scatter(epochs, avg_losses, label='Loss', s=S)
    axs[1, 0].set_title('Loss per Epoch')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].scatter(epochs, calls_per_sec, label='Calls per sec', s=S)
    axs[1, 1].set_title('Calls per sec per Epoch')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Calls per sec')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot various metrics from a CSV file.')
    parser.add_argument('-c', '--csv_filename', type=str, default='averages.csv', help='Path to CSV file containing metrics')
    parser.add_argument('-n', '--ignore_entries', type=int, default=0, help='Number of initial entries to ignore')
    args = parser.parse_args()

    plot_multiple_averages(args.csv_filename, args.ignore_entries)
