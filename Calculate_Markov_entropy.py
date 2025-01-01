import numpy as np
from collections import defaultdict
import os

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