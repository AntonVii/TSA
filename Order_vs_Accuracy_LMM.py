import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Paths
directory = '/Users/mikae/OneDrive/Files/Universitet/EM2/Projekt/TSA_Helper/Analyzer/TSA_Data/Data/DataDecember'
order_sl_path = "/Users/mikae/OneDrive/Files/Universitet/EM2/Projekt/Rapport/OrdersFound.csv"

# Load Orders CSV
order_sl_df = pd.read_csv(order_sl_path)
order_sl_df.rename(columns={"Participant": "Participant_ID", "BestOrder": "MaxOrder"}, inplace=True)

# Function to compute participant accuracy
def compute_accuracy(data, exclude_blocks=None):
    exclude_blocks = exclude_blocks or []
    feedback_data = data[(data['feedback'] == True) & (~data['block_number'].isin(exclude_blocks))]
    accuracy = feedback_data['response_correct'].mean()
    return accuracy

# Function to load data and compute accuracies
def load_accuracies(data_directory, participants):
    accuracies = []
    for participant_id in participants:
        file_path = os.path.join(data_directory, f"Participant_{str(participant_id).zfill(4)}_ExperimentData.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        try:
            # Load participant data
            data = pd.read_csv(file_path)
            
            # Compute accuracy
            accuracy = compute_accuracy(data)
            accuracies.append({'Participant_ID': participant_id, 'Accuracy': accuracy})
        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")
    
    return pd.DataFrame(accuracies)

# Load participant accuracies
participants = order_sl_df['Participant_ID'].unique()
accuracy_df = load_accuracies(directory, participants)

# Merge orders and accuracies
merged_df = pd.merge(order_sl_df[['Participant_ID', 'MaxOrder']], accuracy_df, on='Participant_ID', how='inner')

# Perform correlation analysis
if not merged_df.empty:
    # Pearson correlation
    correlation, p_value = pearsonr(merged_df['MaxOrder'], merged_df['Accuracy'])
    
    # Simple linear regression for R^2 value
    X = sm.add_constant(merged_df['MaxOrder'])
    model = sm.OLS(merged_df['Accuracy'], X).fit()
    r_squared = model.rsquared

    # Display results
    print(f"Correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.3e}")
    print(f"R-squared: {r_squared:.3f}")

    # Plot the relationship
    plt.figure(figsize=(10, 6))
    sns.regplot(x='MaxOrder', y='Accuracy', data=merged_df, scatter_kws={'s': 50}, color='blue')
    plt.title(f"Correlation between Order and Accuracy\n$R^2 = {r_squared:.3f}, p = {p_value:.3e}$", fontsize=14)
    plt.xlabel("Order", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("CorrelationAccuracy.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Mixed Linear Model
    formula = "Accuracy ~ MaxOrder"
    mixed_model = MixedLM.from_formula(
        formula,
        groups=merged_df["Participant_ID"],
        data=merged_df
    ).fit()

    # Print mixed model results
    print(mixed_model.summary())

    # Visualize mixed model predictions
    merged_df['Predicted_Accuracy'] = mixed_model.predict(merged_df)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="MaxOrder", y="Predicted_Accuracy", data=merged_df, color="red", label="Predicted Accuracy")
    sns.scatterplot(x="MaxOrder", y="Accuracy", data=merged_df, color="blue", label="Observed Accuracy", s=50)
    plt.title("Linear Mixed Model for Markov Orders and Accuracy (N=26)", fontsize=14)
    plt.xlabel("Order", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("LMM.png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No data available for analysis.")
