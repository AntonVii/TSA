import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load participant orders from CSV
csv_path = "/Users/mikae/OneDrive/Files/Universitet/EM2/Projekt/Rapport/OrdersFound.csv"  # Replace with the path to your CSV
participant_orders = pd.read_csv(csv_path)

# Ensure participant IDs and orders are loaded correctly
# participant_orders should have columns: ['Participant', 'BestOrder']

# Trial sequences
trialBaseLists = [
    [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 1],
    [1, 0]
]

# Extend sequences to desired lengths
def generate_list(baseList, listLength):
    return (baseList * (listLength // len(baseList) + 1))[:listLength]

trialLists = [generate_list(baseList, 60) for baseList in trialBaseLists]

# Calculate Markov entropy for a given sequence and order
def calculate_entropy(sequence, order):
    from collections import Counter
    import numpy as np

    if order == 0:
        # Calculate Shannon entropy
        counts = Counter(sequence)
        total = len(sequence)
        entropy = 0
        for count in counts.values():
            probability = count / total
            entropy -= probability * np.log2(probability)
        return entropy

    # For higher orders, calculate Markov entropy
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


# Calculate perceived entropy for each participant and sequence
sequence_entropies = []

for idx, sequence in enumerate(trialLists, start=1):  # Sequence indices start at 1
    for _, row in participant_orders.iterrows():
        participant_id = int(float(row['Participant']))  # Convert float to int (e.g., 01.0 -> 1)
  # Format participant ID as "0001", "0002", etc.
        order = int(row['BestOrder'])  # Convert 'BestOrder' to integer
        entropy = calculate_entropy(sequence, order)
        sequence_entropies.append({
            'Sequence': idx,
            'Participant_ID': f"{participant_id} (Order {order})",  # Include order in legend
            'Entropy': entropy
        })

# Convert to DataFrame
entropy_df = pd.DataFrame(sequence_entropies)

# Calculate average entropy for each sequence
average_entropies = entropy_df.groupby('Sequence')['Entropy'].mean().reset_index()

# Adjust the x-axis labels to start from 1
average_entropies['Sequence'] = average_entropies['Sequence'] - 1

# Plot perceived entropies
plt.figure(figsize=(12, 8))
sns.stripplot(
    x='Sequence', y='Entropy', hue='Participant_ID', 
    data=entropy_df, jitter=True, palette='tab20', alpha=0.9
)
plt.plot(average_entropies['Sequence'], average_entropies['Entropy'], color='red', marker='o', label="Average Sub-En")
plt.title("Subjective Entropy of each Trial Sequence as Perceived by Individual Participants")
plt.xlabel("Trial Sequence Number")
plt.ylabel("Subjective Entropy")
plt.legend(title="Participant", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("Perceived_Entropy_Plot_with_Labeled_Legend.png", dpi=600, bbox_inches='tight')  # Save the plot
plt.show()
