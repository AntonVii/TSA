import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

def calculate_entropy(sequence, order):
    """
    Calculate Markov entropy for a given sequence and order.
    """
    if order == 0:
        return 1.0  # Max entropy for zero order (no context)
    
    from collections import defaultdict
    transitions = defaultdict(int)
    context_counts = defaultdict(int)

    for i in range(len(sequence) - order):
        context = tuple(sequence[i:i + order])
        next_symbol = sequence[i + order]
        transitions[(context, next_symbol)] += 1
        context_counts[context] += 1

    entropy = 0
    for context, total_count in context_counts.items():
        context_entropy = 0
        for next_symbol in [0, 1]:
            count = transitions.get((context, next_symbol), 0)
            if count > 0:
                probability = count / total_count
                context_entropy -= probability * np.log2(probability)
        entropy += context_entropy * (total_count / (len(sequence) - order))
    
    return entropy

def generate_random_binary_list(n):
    return [random.choice([0, 1]) for _ in range(n)]

# Predefined thresholds for orders
thresholds = {
    1: 0.7195,
    2: 0.6415,
    3: 0.5365,
    4: 0.3745,
    5: 0.3825,
    6: 0.2165
}

# Parameters for simulation
min_order = 1
max_order = 6
min_len = 10
max_len = 160
nSim = 10000

fSizeLabel = 10
fSizeBoxes = 6

# Initialize results storage
results = []

# Iterate over sequence lengths and orders
for random_sequence_length in range(min_len, max_len + 1):
    for order in range(min_order, max_order + 1):
        threshold = thresholds[order]
        count = 0

        # Simulate sequences and calculate failed percentage
        for sim in range(nSim):
            sequence = generate_random_binary_list(random_sequence_length)
            if calculate_entropy(sequence, order) < threshold:
                count += 1

        failed_percentage = count / nSim
        results.append((random_sequence_length, order, failed_percentage))

# Convert results into a structured array for heatmap
heatmap_data = np.zeros((max_len - min_len + 1, max_order - min_order + 1))

for random_sequence_length, order, failed_percentage in results:
    heatmap_data[random_sequence_length - min_len, order - min_order] = failed_percentage

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_data,
    annot=False,  # Hide numbers in the boxes
    fmt=".2f",
    xticklabels=range(min_order, max_order + 1),
    yticklabels=range(min_len, max_len + 1),  # Provide all y-tick labels
    cmap="YlGnBu"
)

# Customize y-axis ticks to show only every 10th label
plt.yticks(
    ticks=np.arange(0, len(range(min_len, max_len + 1)), 10),  # Indices for every 10th value
    labels=range(min_len, max_len + 1, 10),  # Labels for every 10th value
    fontsize=fSizeLabel  # Adjust font size
)

# Customize x-axis ticks
plt.xticks(fontsize=10)
plt.title(f"Percentage of Sequences below Threshold by Order and Block Length (n = {nSim})")
plt.xlabel("Order")
plt.ylabel("Block Length")
plt.savefig("HeatMapNoLabels.png")
plt.show()
