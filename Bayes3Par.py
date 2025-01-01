from collections import defaultdict
import random
import matplotlib.pyplot as plt
import itertools
import numpy as np
from scipy.optimize import differential_evolution
import csv
import pandas as pd 
from collections import defaultdict
import seaborn as sns
import math
class VOMM:
    def __init__(self, max_order, laplace_smoothing=True, smoothing_value=0.04):
        """
        Initialize the maximum order, context tree, and Laplace smoothing parameters.
        """
        self.max_order = max_order
        self.context_tree = defaultdict(lambda: defaultdict(lambda: smoothing_value))
        self.global_counts = defaultdict(lambda: smoothing_value)  # For 0th-order counts
        self.laplace_smoothing = laplace_smoothing
        self.smoothing_value = smoothing_value

    def train(self, sequence, decay=1, is_interference=False):
        """
        Trains the VOMM by populating the context tree with frequency counts.
        If `is_interference` is True, applies decay to weight interference data.
        """
        for i in range(len(sequence)):
            weight = decay if is_interference else 1  # Apply decay only for interference
            # Update 0th-order counts with weighting
            self.global_counts[sequence[i]] += weight
            for order in range(1, self.max_order + 1):
                if i - order >= 0:
                    # Extract context and the next symbol
                    context = tuple(sequence[i - order:i])
                    next_symbol = sequence[i]
                    # Update the context tree with weighting
                    self.context_tree[context][next_symbol] += weight
        return self.context_tree.items()

    def predict(self, context):
        """
        Predicts the probabilities of symbols (0 and 1).
        Uses a backoff mechanism and proportional smoothing.
        """
        context = tuple(context[-self.max_order:])  # Ensure context length matches max_order

        while context:
            if context in self.context_tree:
                # Get frequency counts for the current context
                symbol_counts = self.context_tree[context]
                total = sum(symbol_counts.values())

                # Apply proportional Laplace smoothing
                if self.laplace_smoothing:
                    # Ensure all possible symbols are included in the smoothing
                    for symbol in [0, 1]:  # Add all possible outcomes
                        if symbol not in symbol_counts:
                            symbol_counts[symbol] = 0

                    total += len(symbol_counts) * self.smoothing_value
                probabilities = {k: (v + self.smoothing_value) / total for k, v in symbol_counts.items()}
                return probabilities.get(0, 0), probabilities.get(1, 0)
            
            # Reduce context length (backoff mechanism)
            context = context[1:]

        # Fall back to 0th-order probabilities
        total_counts = sum(self.global_counts.values())
        if total_counts == 0:
            return 0.5, 0.5  # Default probabilities if no training data

        probabilities = {k: (v + self.smoothing_value) / (total_counts + len(self.global_counts) * self.smoothing_value) 
                         for k, v in self.global_counts.items()}
        return probabilities.get(0, 0), probabilities.get(1, 0)
    def print_context_tree(self):
        """
        Prints the context tree in a readable format.
        """
        print("\n=== Context Tree ===")
        for context, symbol_counts in self.context_tree.items():
            formatted_counts = ', '.join([f"{symbol}: {count:.4f}" for symbol, count in symbol_counts.items()])
            print(f"Context: {context} -> {formatted_counts}")
        print("\n=== Global Counts (0th Order) ===")
        formatted_global = ', '.join([f"{symbol}: {count:.4f}" for symbol, count in self.global_counts.items()])
        print(f"Global Counts: {formatted_global}")


def generate_correctness_list_with_overlap(data_list, participant_response, interference, max_order, decay,smooth, block_size):
    correctness_list = []

    # Split the data into blocks
    num_blocks = len(data_list) // block_size
    blocks = [data_list[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]
    pa_answer_block = [participant_response[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]

    # Iterate through each block
    for block_index, block in enumerate(blocks):
        # Initialize the model for each block
        model = VOMM(max_order,smoothing_value=smooth)
        # Train on the interference with a uniform decay
        model.train(interference[block_index * 20:(block_index + 1) * 20], decay=decay, is_interference=True)
        
        # Iterate through each element in the block
        for i in range(len(block)):
            sequence = block[i - max_order - 1:i]
            model.train(sequence)

            # Ensure we're accessing the corresponding participant response
            next_value = pa_answer_block[block_index][i]
            
            probabilities = model.predict(sequence)

            # Replace zero probabilities before appending to the correctness_list
            if probabilities[next_value] == 0.0:
                probabilities = (0.001, 0.999) if next_value == 0 else (0.999, 0.001)

            correctness_list.append(((next_value, probabilities[next_value]), probabilities))  # Access probability directly

    return correctness_list





# Function to generate list based on base list and length
def generate_list(base_list, list_length):
    cut_list = (base_list * (int(list_length)))[0:list_length]
    return cut_list

# Function to plot accuracy over time without reset, but divided into blocks, with color change for each block
def plot_accuracy_with_overlap(data_list,reference_list,inteference, maxorder,decay ,smooth,block_size):
    guesses = generate_correctness_list_with_overlap(data_list,reference_list,inteference, maxorder,decay ,smooth,block_size)
    guesses = [random.choices([0,1],weights = [i,k])[0] for (_,(i,k)) in guesses]
    guesses = [1 if guess == actual else 0 for guess, actual in zip(guesses, data_list)]
    cumulative_accuracies = []
    cumulative_accuraciesREF=[]
    start_index = 0
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  # Cycle through different colors

    # Calculate cumulative accuracy for each block separately and plot as a single continuous graph
    for i in range(0, len(guesses), block_size):
        # Each block may overlap by `sequence_length` with the next
        correctness_list = guesses[i:i + block_size]
        cumulative_accuracy = [sum(correctness_list[:i + 1]) / (i + 1) for i in range(len(correctness_list))]
        cumulative_accuracies.extend(cumulative_accuracy)

        # Plot segment for the current block with different color
        block_indices = range(start_index, start_index + len(cumulative_accuracy))
        plt.plot(block_indices, cumulative_accuracy, label=f"Block {i // (block_size) + 1}", linestyle='-', color="black")
        plt.gca().set_facecolor('lightgrey' if (i // (block_size)) % 2 == 0 else 'white')
        
        start_index += len(cumulative_accuracy)
        

    reference_list = [1 if guess == actual else 0 for guess, actual in zip(reference_list, data_list)]
    start_indexREF = 0
    for i in range(0, len(reference_list), block_size):
        # Each block may overlap by `sequence_length` with the next
        correctness_listREF = reference_list[i:i + block_size]
        cumulative_accuracyREF = [sum(correctness_listREF[:i + 1]) / (i + 1) for i in range(len(correctness_listREF))]
        cumulative_accuraciesREF.extend(cumulative_accuracyREF)

        # Plot segment for the current block with different color
        block_indices = range(start_indexREF, start_indexREF + len(cumulative_accuracyREF))
        plt.plot(block_indices, cumulative_accuracyREF, label=f"Block {i // (block_size) + 1}", linestyle='-', color=next(colors))
        plt.gca().set_facecolor('lightgrey' if (i // (block_size)) % 2 == 0 else 'white')
        start_indexREF += len(cumulative_accuracyREF)
        
    # Finalize the plot
    plt.xlabel("Index")
    plt.ylabel("Accuracy")
    plt.title("Prediction Accuracy Over Time with Overlap (Continuous Across Blocks)")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.gcf().set_size_inches(13, 6)
    plt.show()


# Example trial base lists
trialBaseLists = [
    [0,1,0,1,1,0,0,0,1,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,0,0,0,1,1,0,1,1,1],
    [0,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,1,1,0,0,1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,0],
    [1,1,0,0,1,1],
    [1,0,0,0,1,0,1,1,1,0],
    [0,0,0,0,1,0,0,1,1,0,0,1],
    [1,0,1,1,1,1,0,0,0,1,0,1],
    [0,1,0,1,0,1,1,0,1,0,1,0],
    [0,0,0,0,1],
    [0,1,1],
    [1,0]
]
trialListLength = 60

# Generate trial lists and flatten them into total_run
trial_lists = [generate_list(base_list, trialListLength) for base_list in trialBaseLists]
trial_list_combined = [item for sublist in trial_lists for item in sublist]

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_points_with_sizes(data):
    """
    Creates a 3D scatter plot from a list of tuples with four elements.
    The first three elements represent coordinates, and the fourth is the size of the point.

    Args:
        data (list of tuples): A list where each tuple contains four elements 
                               (x, y, z, size).
    """
    # Unpack the tuples into separate lists
    x_coords, y_coords, z_coords, sizes = zip(*data)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(13,7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_coords, y_coords, z_coords, s=[i * 1000 for i in sizes])

    # Add labels
    ax.set_xlabel('order')
    ax.set_ylabel('decay')
    ax.set_zlabel('smooth')

    # Show the plot
    plt.show()

def get_trials_with_feedback(file_path):
    data = pd.read_csv(file_path)

    # Filter the data into two separate datasets: with feedback and without feedback
    trials_with_feedback = data[data['feedback'] == True].copy()
    trials_with_no_feedback = data[data['feedback'] == False].copy()

    # Convert 'response_correct' to lowercase and remove spaces for both sets
    trials_with_feedback.loc[:, 'participant_response'] = trials_with_feedback['participant_response'].astype(str).str.lower().str.strip()
    trials_with_no_feedback.loc[:, 'participant_response'] = trials_with_no_feedback['participant_response'].astype(str).str.lower().str.strip()

    # Convert 'response_correct' to binary values: 1 for 'l' and 0 for 's'
    binary_responses_feedback = trials_with_feedback['participant_response'].apply(lambda x: 1 if x == 's' else 0).astype(int).tolist()
    binary_responses_no_feedback = trials_with_no_feedback['participant_response'].apply(lambda x: 1 if x == 's' else 0).astype(int).tolist()

    return binary_responses_feedback, binary_responses_no_feedback


def brute_force_bayes_factors(base_path, num_participants, max_order, block_size):
    results = []  # To store results for all participants

    # Define decay and smooth values
    decay_values = [round(i * 0.1, 2) for i in range(1, 11)]  # Decay from 1.0 to 0.1
    decay_values.reverse()
    smooth_values = [round(i * 0.1, 2) for i in range(1, 11)]  # Smooth from 1.0 to 0.1
    smooth_values.reverse()

    for participant_id in range(1, num_participants + 1):
        participant_id_str = f"{participant_id:04d}"
        file_path = f"{base_path}{participant_id_str}_ExperimentData.csv"

        participant_result = {
            "Participant": participant_id,
            "BestDecay": None,
            "BestOrder": None,
            "BestSmooth": None,
            "BestPosterior": None,
            "MeanDecay": None,
            "MeanOrder": None,
            "MeanSmooth": None,
            "VarianceDecay": None,
            "VarianceOrder": None,
            "VarianceSmooth": None,
            "MedianDecay": None,
            "MedianOrder": None,
            "MedianSmooth": None,
            "BayesFactor_Best_vs_Null": None,
            "BayesFactor_Total_vs_Null": None,
        }

        try:
            # Load participant's feedback and no-feedback trial data
            feedback_trials_list, no_feedback_list = get_trials_with_feedback(file_path)

            # Initialize variables
            prior_nul = 0.5
            priorModel = 1 / (len(decay_values) * max_order * len(smooth_values)) * (1-prior_nul)
            priorRandom = prior_nul
            random_likelihood = 0.5 ** 600
            model_likelihoods = []

            # Iterate over decay, smooth, and order to calculate likelihoods
            for smooth in smooth_values:
                for decay in decay_values:
                    for order in range(0, max_order + 1):
                        likelihood_list = generate_correctness_list_with_overlap(
                            trial_list_combined,
                            feedback_trials_list,
                            no_feedback_list,
                            order,
                            decay,
                            smooth,
                            block_size,
                        )
                        total_likelihood = total(likelihood_list) * 100
                        model_likelihoods.append((order, decay, smooth, total_likelihood))

            # Compute the posterior probabilities
            data_likelihood = random_likelihood * priorRandom + sum(
                [priorModel * likelihood for order, decay, smooth, likelihood in model_likelihoods]
            )
            posterior = [(0, 0, 0, random_likelihood * priorRandom / data_likelihood)] + [
                (order, decay, smooth, priorModel * likelihood / data_likelihood)
                for order, decay, smooth, likelihood in model_likelihoods
            ]

            # Extract mode posterior (highest posterior value)
            mode_posterior = max(posterior[1:], key=lambda x: x[3])
            mode_order, mode_decay, mode_smooth, mode_posterior_value = mode_posterior

            # Initialize variables for mean, variance, and median calculation
            mean_order, mean_decay, mean_smooth = 0, 0, 0
            variance_order, variance_decay, variance_smooth = 0, 0, 0
            posterior_values = [p[3] for p in posterior[1:]]
            marginal_orders = [p[0] for p in posterior[1:]]
            marginal_decays = [p[1] for p in posterior[1:]]
            marginal_smooths = [p[2] for p in posterior[1:]]

            # Calculate mean
            for order, decay, smooth, posterior_value in posterior[1:]:
                mean_order += order * posterior_value
                mean_decay += decay * posterior_value
                mean_smooth += smooth * posterior_value

            # Calculate variance
            for order, decay, smooth, posterior_value in posterior[1:]:
                variance_order += posterior_value * ((order - mean_order) ** 2)
                variance_decay += posterior_value * ((decay - mean_decay) ** 2)
                variance_smooth += posterior_value * ((smooth - mean_smooth) ** 2)

            # Calculate median
            def weighted_median(values, weights):
                if len(values) == 0 or len(weights) == 0:
                    raise ValueError("Input values or weights are empty.")
                
                sorted_indices = np.argsort(values)
                sorted_values = np.array(values)[sorted_indices]
                sorted_weights = np.array(weights)[sorted_indices]
                
                cumulative_weights = np.cumsum(sorted_weights)
                median_indices = np.where(cumulative_weights >= 0.5)[0]
                
                # If no valid median index is found, return 0
                if len(median_indices) == 0:
                    return 0
                
                median_idx = median_indices[0]
                return sorted_values[median_idx]

            median_order = weighted_median(marginal_orders, posterior_values)
            median_decay = weighted_median(marginal_decays, posterior_values)
            median_smooth = weighted_median(marginal_smooths, posterior_values)

            # Calculate Bayes Factors
            best_model_likelihood = max([likelihood for _, _, _, likelihood in model_likelihoods])
            total_model_likelihood = sum([likelihood for _, _, _, likelihood in model_likelihoods])

            shannon_total_model_likelihood = sum([likelihood*priorModel for order, _, _, likelihood in model_likelihoods if order == 0])

            markov_total_model_likelihood = sum([likelihood*priorModel for order, _, _, likelihood in model_likelihoods if order != 0])

            markovBF = markov_total_model_likelihood/shannon_total_model_likelihood
            bayes_factor_best_vs_null = best_model_likelihood / random_likelihood
            bayes_factor_total_vs_null = sum([p*priorModel for _,_,_,p in posterior[1:]]) / (posterior[0])[3]
            if participant_id == 2:
                plot_accuracy_with_overlap(trial_list_combined,feedback_trials_list,no_feedback_list, mode_order,mode_decay ,mode_smooth,60)
            # Update participant result
            participant_result.update(
                {
                    "BestDecay": mode_decay,
                    "BestOrder": mode_order,
                    "BestSmooth": mode_smooth,
                    "BestPosterior": mode_posterior_value,
                    "MeanDecay": mean_decay,
                    "MeanOrder": mean_order,
                    "MeanSmooth": mean_smooth,
                    "VarianceDecay": variance_decay,
                    "VarianceOrder": variance_order,
                    "VarianceSmooth": variance_smooth,
                    "MedianDecay": median_decay,
                    "MedianOrder": median_order,
                    "MedianSmooth": median_smooth,
                    "BayesFactor_Best_vs_Null": bayes_factor_best_vs_null,
                    "BayesFactor_Shannon_vs_Markov": markovBF,
                    "BayesFactor_Total_vs_Null": bayes_factor_total_vs_null,
                    "posterior_Null" :([p for _,_,_,p in posterior])[0],
                    "posterior_0_order" : shannon_total_model_likelihood
                }
            )

        except Exception as e:
            print(f"Error processing Participant {participant_id_str}: {e}")
            continue

        # Append results for the current participant
        results.append(participant_result)

    csv_file_path = "/Users/Anton/Desktop/TSA/BayesBased/BayesPosteriors3.csv"
    with open(csv_file_path, mode='w', newline='') as csv_file:
        # Define the fields to include in the CSV
        fieldnames = [
            "Participant",
            "BestDecay", "BestOrder", "BestSmooth", "BestPosterior",
            "MeanDecay", "MeanOrder", "MeanSmooth",
            "VarianceDecay", "VarianceOrder", "VarianceSmooth",
            "MedianDecay", "MedianOrder", "MedianSmooth",
            "BayesFactor_Best_vs_Null","BayesFactor_Shannon_vs_Markov", "BayesFactor_Total_vs_Null",
            "posterior_Null", "posterior_0_order",
        ]

        # Create the CSV writer
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the results for each participant
        writer.writerows(results)

    print(f"Results saved to {csv_file_path}")




# Define the total function for likelihood computation
def total(likelyhood_list):
    likelyhood_total = 1
    for ((_, likelihood), _) in likelyhood_list:
        likelyhood_total *= likelihood
    return likelyhood_total

# Run the brute force analysis
if __name__ == "__main__":
    base_path = "/Users/Anton/Desktop/TSA/Data/Participant_"
    num_participants = 26  # Adjust the number of participants as needed
    max_order = 8  # Change this value as needed
    block_size = 60
    decay = 1

    brute_force_bayes_factors(base_path, num_participants, max_order, block_size)


import pandas as pd
def plot_max_order_distribution_bar(results):
    # Extract the 'BestOrder' values from the results
    max_orders = results['BestOrder']
    BF = zip(results["BayesFactor_Shannon_vs_Markov"],results["BayesFactor_Best_vs_Null"], results["BayesFactor_Total_vs_Null"])

    SM_total = 0
    BN_total = 0
    TN_total = 0

    for SM,BN,TN in BF:
        if SM and BN and TN != 0:
            SM_total+=math.log10(SM)
            BN_total+=math.log10(BN)
            TN_total+=math.log10(TN)
        

    print(f"The BF for shannon vs Markov {SM_total} The BF for Best mode vs Null is {BN_total}, The BF for All models vs Null {TN_total} ")

    max_orders = [0 if i is None else i for i in max_orders]

    # Count the frequency of each 'MaxOrder' value
    order_counts = pd.Series(max_orders).value_counts().sort_index()

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    order_counts.plot(kind='bar', color='skyblue', edgecolor='black', alpha=0.7)

    # Set the labels and title
    plt.xlabel("Max Order")
    plt.ylabel("Frequency")
    plt.title("Distribution of Max Orders Across Participants")

    # Set the x-axis ticks to display integer values
    plt.xticks(rotation=0)

    # Display the plot
    plt.tight_layout()
    plt.show()

# Path to the CSV file
csv_file_path = "/Users/Anton/Desktop/TSA/BayesBased/BayesPosteriors3.csv"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Call the function to plot the bar plot
plot_max_order_distribution_bar(df)

