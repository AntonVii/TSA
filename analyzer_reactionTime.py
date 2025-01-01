import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load your data
participant_ID = '0020'
file_path = '/Users/mikae/OneDrive/Files/Universitet/EM2/Projekt/TSA_Helper/Analyzer/TSA_Data/Participant_'+participant_ID+'_ExperimentData.csv'
df = pd.read_csv(file_path)

# Plot reaction_time per trial as a line diagram, using row index as x-axis
plt.plot(df.index, df['reaction_time'], label='Reaction Time', color='blue', alpha=0.5)

# Calculate the running average of reaction time using a rolling window
window_size = 10  # Adjust for smoothing effect
df['reaction_time_running_avg'] = df['reaction_time'].rolling(window=window_size, min_periods=1).mean()

# Plot the running average of reaction time as a smooth line
plt.plot(df.index, df['reaction_time_running_avg'], label='Running Average of Reaction Time', color='red', linewidth=2)

# Add shaded regions for each block and the block average lines
current_block_start = 0
block_avgs = []  # To store the average reaction time of each block
block_indices = []  # To store the block index for the trend line

for i in range(1, len(df)):
    if df['trial_number'][i] < df['trial_number'][i - 1]:  # Detects the start of a new block
        # Determine block type for shading (e.g., 'False' = no feedback, 'True' = feedback)
        color = 'lightgrey' if not df['feedback'][current_block_start] else 'white'
        plt.axvspan(current_block_start, i - 1, facecolor=color, alpha=0.3)

        # Calculate and plot the average reaction time for the current block
        block_avg = df['reaction_time'][current_block_start:i].mean()
        block_avgs.append(block_avg)
        block_indices.append(current_block_start)  # Store start index for trend line

        # Plot the horizontal line only within the current block range
        plt.plot([current_block_start, i - 1], [block_avg, block_avg], color='black', linestyle='--')
        
        current_block_start = i

# Calculate and plot the average reaction time for the last block if needed
block_avg = df['reaction_time'][current_block_start:].mean()
block_avgs.append(block_avg)
block_indices.append(current_block_start)
plt.plot([current_block_start, len(df) - 1], [block_avg, block_avg], color='black', linestyle='--')

# Fit a linear regression line to the block averages
model = LinearRegression()
block_indices_reshaped = [[idx] for idx in block_indices]  # Reshape for sklearn
model.fit(block_indices_reshaped, block_avgs)
trend_line = model.predict(block_indices_reshaped)

# Plot the trend line for the block averages
plt.plot(block_indices, trend_line, label='Trend Line (Block Averages)', color='green', linestyle='-', linewidth=2)

# Add labels and legend
plt.xlabel('Trial')
plt.ylabel('Reaction Time')
plt.title('Reaction Time per Trial (FP: '+participant_ID+')')
plt.legend()

# Show plot
plt.show()
