import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load participant orders from CSV
csv_path = "/Users/mikae/OneDrive/Files/Universitet/EM2/Projekt/Rapport/OrdersFound.csv"  # Replace with the path to your CSV
participant_orders = pd.read_csv(csv_path)

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
        participant_id = int(float(row['Participant']))  # Convert float to int
        order = int(row['BestOrder'])  # Convert 'BestOrder' to integer
        entropy = calculate_entropy(sequence, order)
        sequence_entropies.append({
            'Sequence': idx,
            'Participant_ID': participant_id,
            'Order': order,
            'Entropy': entropy
        })

# Convert to DataFrame
entropy_df = pd.DataFrame(sequence_entropies)

# Group by Sequence and Order to calculate averages
grouped_entropy = (
    entropy_df.groupby(['Sequence', 'Order'])['Entropy']
    .agg(['mean', 'count'])  # Calculate mean entropy and count participants
    .reset_index()
    .rename(columns={'mean': 'Average_Entropy', 'count': 'Participant_Count'})
)

# Add jitter to x-axis for each sequence
jitter_strength = 0.1
grouped_entropy['Jittered_Sequence'] = grouped_entropy['Sequence'] + np.random.uniform(
    -jitter_strength, jitter_strength, size=len(grouped_entropy)
)

# Calculate overall averages per sequence
average_entropies = entropy_df.groupby('Sequence')['Entropy'].mean().reset_index()

# Plot perceived entropies with jitter
plt.figure(figsize=(14, 8))

# Plot average entropy as a red line
plt.plot(average_entropies['Sequence'], average_entropies['Entropy'], color='red', marker='o', label='Average Sub-En')

# Plot group-level averages (6 points per sequence) with jitter
sns.scatterplot(
    x='Jittered_Sequence', y='Average_Entropy', hue='Order',
    data=grouped_entropy, palette='tab10', legend='full', s=100, alpha=0.8
)

plt.xticks(ticks=range(1, len(trialLists) + 1), labels=range(1, len(trialLists) + 1))

# Title and labels
plt.title("Subjective Entropy of each Trial Sequence as Pereceived by each Markov Order", fontsize=16)
plt.xlabel("Sequence Number", fontsize=12)
plt.ylabel("Subjective Entropy", fontsize=12)
plt.legend(title="Order", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("Perceived_Entropy_Grouped_with_Jitter.png", dpi=300, bbox_inches='tight')
plt.show()
