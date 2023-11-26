import argparse
import csv
import matplotlib.pyplot as plt

def plot_averages(csv_filename, ignore_entries=0):
    epochs = []
    avg_q_values = []
    avg_rewards = []

    with open(csv_filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(csv_reader):
            if i < ignore_entries:
                continue  # Skip the initial rows

            epochs.append(int(row['Epoch']))
            avg_q_values.append(float(row['Average_Q_value']))
            avg_rewards.append(float(row['Average_Reward']))

    # Plotting the averages
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_q_values, label='Average Q-value')
    plt.plot(epochs, avg_rewards, label='Average Reward')
    plt.title('Average Q-value and Average Reward per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot average Q-values and rewards from a CSV file.')
    parser.add_argument('-c', '--csv_filename', type=str, default='averages.csv', help='Path to CSV file containing averages')
    parser.add_argument('-n', '--ignore_entries', type=int, default=0, help='Number of initial entries to ignore')
    args = parser.parse_args()

    plot_averages(args.csv_filename, args.ignore_entries)
