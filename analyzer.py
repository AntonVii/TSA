import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate running average accuracy
def calculate_running_avg(sub_df):
    sub_df['running_avg_accuracy'] = sub_df['response_correct'].expanding().mean()
    return sub_df

# Function to create plot for a single participant
def plot_running_avg_accuracy(data, save_path='running_avg_accuracy_per_trial_plot_with_blocks.jpg'):
    # Ensure response_correct is interpreted as a boolean
    data['response_correct'] = data['response_correct'].astype(bool)

    # Reset index to avoid ambiguity with columns
    data = data.reset_index(drop=True)

    # Convert 'camera_on' to 1 for True and 0 for False
    data['camera_on'] = data['camera_on'].astype(int)

    # Calculate a running average of accuracy per trial for each block
    # Adjust trial number to be cumulative across blocks
    data['cumulative_trial_number'] = data.groupby('block_number').cumcount() + 1

    # Offset trial numbers to make them cumulative across all blocks
    cumulative_offset = 0
    for block_num in sorted(data['block_number'].unique()):
        data.loc[data['block_number'] == block_num, 'cumulative_trial_number'] += cumulative_offset
        cumulative_offset = data.loc[data['block_number'] == block_num, 'cumulative_trial_number'].max()

    # Apply calculate_running_avg without selecting 'response_correct' explicitly
    data = data.groupby('block_number', as_index=False).apply(calculate_running_avg)

    # Calculate whole-block accuracy for each block
    block_accuracies = data.groupby('block_number')['response_correct'].mean()

    # Calculate expected accuracy for each block based on 'camera_on' distribution
    expected_accuracies = {}
    for block_num, block_data in data.groupby('block_number'):
        num_ones = block_data['camera_on'].sum()
        num_zeros = len(block_data) - num_ones
        prob_1 = num_ones / len(block_data)  # Probability of 1's in the block
        prob_0 = num_zeros / len(block_data)  # Probability of 0's in the block

        # Expected accuracy with guessing "1" prob_1 of the time and "0" prob_0 of the time
        expected_accuracy = (num_ones * prob_1) + (num_zeros * prob_0)
        expected_accuracies[block_num] = expected_accuracy / len(block_data)

    # Plot the running average accuracy per trial with background color for each block
    plt.figure(figsize=(13, 6))
    block_edges = []

    for i, (block_num, block_data) in enumerate(data.groupby('block_number')):
        # Shade blocks
        plt.axvspan(block_data['cumulative_trial_number'].iloc[0], block_data['cumulative_trial_number'].iloc[-1], color='lightgray' if i % 2 == 0 else 'whitesmoke', alpha=0.3)

        # Plot running average accuracy
        plt.plot(block_data['cumulative_trial_number'], block_data['running_avg_accuracy'], linestyle='-', label=f'Block {block_num}')

        # Add whole-block accuracy line without adding it to the legend
        plt.hlines(y=block_accuracies[block_num], xmin=block_data['cumulative_trial_number'].iloc[0],
                   xmax=block_data['cumulative_trial_number'].iloc[-1], colors='red', linestyle='--')

        # Add expected accuracy line without adding it to the legend
        plt.hlines(y=expected_accuracies[block_num], xmin=block_data['cumulative_trial_number'].iloc[0],
                   xmax=block_data['cumulative_trial_number'].iloc[-1], colors='blue', linestyle=':', linewidth=2)

        # Record block edges for labeling
        block_edges.append(block_data['cumulative_trial_number'].iloc[-1])

    # Add general lines to the legend for the accuracy types
    plt.plot([], [], color='red', linestyle='--', label='Whole-block Accuracy')
    plt.plot([], [], color='blue', linestyle=':', linewidth=2, label='Expected Accuracy')

    # Adjust legend to be outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Add block number labels at the bottom
    for i, block_num in enumerate(data['block_number'].unique()):
        # Get the middle of the block range
        x_position = (data[data['block_number'] == block_num]['cumulative_trial_number'].iloc[0] + block_edges[i]) / 2
        plt.text(x_position, -0.05, f'B{block_num}', ha='center', va='top', fontsize=10, color='black')

    plt.title('Running Average Accuracy per Trial (Restarted for Each Block)')
    plt.xlabel('Cumulative Trial Number')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()  # Ensure everything fits within the figure

    # Save and display the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

# Example usage
if __name__ == "__main__":
    # Load the dataset
    file_path = '/Users/Anton/Desktop/TSA/Data/Participant_0011_ExperimentData.csv'
    data = pd.read_csv(file_path)
    plot_running_avg_accuracy(data)