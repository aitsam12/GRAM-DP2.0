
from GRAM_DP2_experiments_main import DifferentialPrivacyLibrary
import matplotlib.pyplot as plt
import pandas as pd

# Example usage
dataset_paths = ["Sleep.csv", "adult.csv", "diabetes.csv"]  # Replace with your actual file paths
column_name = "Age"
query_types = ['count', 'sum', 'mean', 'variance']
privacy_levels = ["very_high", "high", "moderate", "low", "very_low"]
num_iterations = 50
y_scale = (0.1, 10)  # Set the y-axis scale

# Adjusting figure size to prevent overlapping
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Iterate over each query type
for query_type_idx, query_type in enumerate(query_types):
    
    # Plot results for each dataset
    for dataset_path in dataset_paths:
        average_results = []
        dataset = pd.read_csv(dataset_path)  # Assuming the file is in CSV format
        dataset_length = len(dataset)
        
        # Iterate over each privacy level
        for level in privacy_levels:
            total_epsilon = 0

            # Run the differential privacy algorithm 100 times
            for _ in range(num_iterations):
                privacy_library = DifferentialPrivacyLibrary(dataset_path, column_name, query_type, level)
                resulting_epsilon = privacy_library.run_differential_privacy()
                total_epsilon += resulting_epsilon

            # Calculate the average epsilon value
            average_epsilon = total_epsilon / num_iterations
            average_results.append(average_epsilon)

        # Plotting the average results for each dataset
        axs[query_type_idx].plot(privacy_levels, average_results, marker='o', label=f'Dataset {dataset_paths.index(dataset_path)+1} (len={dataset_length})')
        axs[query_type_idx].set_title(f" {query_type.capitalize()} Query", size=16)
        axs[query_type_idx].grid(True)
        axs[2].tick_params(axis='x', labelsize=14)
        axs[0].tick_params(axis='y', labelsize=14)  
        axs[2].tick_params(axis='y', labelsize=14) 

        # Adding legend only to the first subplot
        if query_type_idx == 0:
            axs[query_type_idx].legend(prop={'size': 13})

# Set the same y-axis scale for all subplots
for ax in axs:
    ax.set_ylim(y_scale)

# Labeling the shared x-axis and y-axis
fig.text(0.015, 0.5, 'Average Epsilon Value', va='center', rotation='vertical', size=16)  # Positioning the y-label in the middle of the left side
plt.xticks(fontsize=14)

plt.tight_layout(pad=2.5)  # Adjust layout with padding
plt.show()
