import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from scipy import stats

from sklearn.metrics import r2_score


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


def process_participant(file_path, order, feedback_blocks):
    data = pd.read_csv(file_path)
    results = []
    for block in feedback_blocks:
        block_data = data[data['block_number'] == block]
        sequence = block_data['correct_response'].map({'s': 1, 'l': 0}).tolist()
        if len(sequence) < order + 1:
            continue
        entropy = calculate_entropy(sequence, order)
        accuracy = block_data['response_correct'].mean()
        p_ones = sum(sequence) / len(sequence)
        expected_accuracy = max(p_ones, 1 - p_ones)
        results.append({'Sequence': block, 'Entropy': entropy, 'Accuracy': accuracy, 'Expected_Accuracy': expected_accuracy, 'BestOrder': order})
    return results


def aggregate_results(directory, order_sl_df, feedback_blocks):
    results = []
    for _, row in order_sl_df.iterrows():
        participant_id = int(row['Participant'])
        order = int(row['BestOrder'])
        file_path = f"{directory}/Participant_{str(participant_id).zfill(4)}_ExperimentData.csv"
        try:
            participant_results = process_participant(file_path, order, feedback_blocks)
            results.extend(participant_results)
        except FileNotFoundError:
            print(f"Data for Participant {participant_id} not found.")
    return pd.DataFrame(results)


def fit_regression_lines(results_df):
    regression_models = {}
    regression_significance = {}
    for order, group in results_df.groupby('BestOrder'):
        X = group['Entropy'].values.reshape(-1, 1)
        y_actual = group['Accuracy'].values

        # Fit the regression model
        model_actual = LinearRegression().fit(X, y_actual)

        # Calculate R^2, F-statistic, and p-value
        y_pred = model_actual.predict(X)
        r_squared = r2_score(y_actual, y_pred)
        n = len(y_actual)  # Number of observations
        k = 1  # Number of predictors (Entropy)
        f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
        p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)  # F-test p-value

        # Store the regression model and significance results
        regression_models[order] = model_actual
        regression_significance[order] = {'F-statistic': f_stat, 'p-value': p_value}

    return regression_models, regression_significance



def plot_results(results_df, regression_models, thresholds, selected_orders):
    plt.figure(figsize=(12, 8))
    for order in selected_orders:
        if order not in regression_models:
            print(f"Order {order} not found in the dataset.")
            continue

        group = results_df[results_df['BestOrder'] == order]
        X = np.linspace(0, results_df['Entropy'].max(), 500).reshape(-1, 1)

        # Actual values model
        model_actual = regression_models[order]
        y_actual = model_actual.predict(X)

        # Expected accuracy (mean-based line)
        expected_accuracy = group['Expected_Accuracy'].mean()
        y_expected = np.full_like(X.flatten(), expected_accuracy)

        # Plot points and regression lines
        plt.scatter(group['Entropy'], group['Accuracy'], label=f"Order {order} Actual", alpha=0.6)
        plt.scatter(group['Entropy'], group['Expected_Accuracy'], label=f"Order {order} Expected", alpha=0.6)
        plt.plot(X, y_actual, label=f"Order {order} Actual Fit", linestyle='--', color='blue')
        plt.plot(X, y_expected, label=f"Order {order} Expected Fit", linestyle='--', color='green')

        # Add threshold line if found
        if thresholds.get(order):
            plt.axvline(x=thresholds[order], color='red', linestyle='-', label=f"Threshold (Order {order})")

    plt.xlabel("Subjective Entropy")
    plt.ylabel("Accuracy over all feedback-blocks")
    plt.title("Entropic Threshold by Order")
    plt.legend()
    plt.show()




def sliding_window_significance(results_df, regression_models, selected_orders, window_size=0.005, step_size=0.001):
    thresholds = {}
    print("\nSliding Window Test Results by Order:")

    # Fixed entropy range [0, 1]
    entropy_range = (0, 1)

    for order in selected_orders:
        if order not in regression_models:
            print(f"Order {order} not found in the dataset.")
            continue

        model_actual = regression_models[order]  # Use only the actual values model
        significant_threshold = None
        last_significant = None

        print(f"\nOrder {order}:")

        for start in np.arange(entropy_range[1] - window_size, entropy_range[0] - step_size, -step_size):
            end = start + window_size
            mid_point = (start + end) / 2

            # Predict accuracies for actual using regression model
            X_window = np.linspace(start, end, 10).reshape(-1, 1)
            predicted_actual = model_actual.predict(X_window)

            # Calculate the expected accuracy as the mean
            group = results_df[results_df['BestOrder'] == order]
            if group.empty:
                print(f"No data available for Order {order}")
                break
            expected_accuracy = group['Expected_Accuracy'].mean()
            predicted_expected = np.full_like(predicted_actual, expected_accuracy)

            # Perform paired t-test
            t_stat, p_value = ttest_rel(predicted_actual, predicted_expected)

            # Check significance
            if p_value >= 0.05:
                significant_threshold = mid_point
                # Print information for the current non-significant test and the last significant test
                if last_significant is not None:
                    print(f"Last Significant: {last_significant['start']:.4f}-{last_significant['end']:.4f} | "
                          f"t-Statistic: {last_significant['t_stat']:.4f} | "
                          f"p-Value: {last_significant['p_value']:.4f}")
                print(f"First Non-Significant: {start:.4f}-{end:.4f} | t-Statistic: {t_stat:.4f} | p-Value: {p_value:.4f}")
                break
            else:
                # Save the last significant test details
                last_significant = {
                    'start': start,
                    'end': end,
                    't_stat': t_stat,
                    'p_value': p_value
                }

        thresholds[order] = significant_threshold
        if significant_threshold is not None:
            print(f"Threshold for Order {order}: {significant_threshold:.6f}")
        else:
            print(f"Threshold for Order {order}: Always significant across the range.")

    return thresholds



def plot_thresholds(thresholds, selected_orders):
    # Prepare data for plotting
    orders = []
    threshold_values = []
    for order in selected_orders:
        if thresholds.get(order) is not None:
            orders.append(order)
            threshold_values.append(thresholds[order])

    # Convert to arrays for regression
    X = np.array(orders).reshape(-1, 1)  # Independent variable (orders)
    y = np.array(threshold_values)  # Dependent variable (thresholds)

    # Fit linear regression model
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Regression coefficients
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)  # R^2 value
    equation = f"y = {slope:.3f}x + {intercept:.3f}, R² = {r_squared:.3f}"

    # Plot thresholds as data points
    plt.figure(figsize=(10, 6))
    plt.scatter(orders, threshold_values, color='blue', label="Threshold", s=100)

    # Plot regression line
    X_line = np.linspace(min(orders), max(orders), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linestyle='-', label=f"Regression Line\n{equation}")

    # Plot settings
    plt.xlabel("Order")
    plt.ylabel("Entropic Threshold")
    plt.title("Entropic Thresholds by Order")
    plt.xticks(orders)
    plt.ylim(0, 1)  # Assuming entropy is always between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()





def main():
    # File paths
    directory = '/Users/mikae/OneDrive/Files/Universitet/EM2/Projekt/TSA_Helper/Analyzer/TSA_Data/Data/DataDecember'
    order_sl_path = '/Users/mikae/OneDrive/Files/Universitet/EM2/Projekt/Rapport/OrdersFound.csv'
    order_sl_df = pd.read_csv(order_sl_path)
    feedback_blocks = range(2, 21, 2)

    # User-defined orders to analyze
    selected_orders = [1,2,3,4,5,6]  # Modify this list to include the orders you want to analyze

    # Aggregate results
    results_df = aggregate_results(directory, order_sl_df, feedback_blocks)

    # Fit regression models and test significance
    regression_models, regression_significance = fit_regression_lines(results_df)

    # Print regression significance results only for selected orders
    print("\nRegression Significance Test by Order:")
    for order in selected_orders:
        if order in regression_significance:
            stats = regression_significance[order]
            print(f"Order {order}: F-statistic = {stats['F-statistic']:.4f}, p-value = {stats['p-value']:.4e}")
            if stats['p-value'] < 0.05:
                print(f"Order {order}: Regression is significant.")
            else:
                print(f"Order {order}: Regression is not significant.")
        else:
            print(f"Order {order}: No regression results available.")

    # Perform sliding window significance testing for selected orders
    thresholds = sliding_window_significance(results_df, regression_models, selected_orders)

    # Plot results for selected orders
    plot_results(results_df, regression_models, thresholds, selected_orders)

    # Plot entropic thresholds
    plot_thresholds(thresholds, selected_orders)

    
if __name__ == "__main__":
    main()
