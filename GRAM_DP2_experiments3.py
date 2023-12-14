

from GRAM_DP2_experiments_main import DifferentialPrivacyLibrary
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics

# Initialize variables
dataset_paths = ["Sleep.csv", "adult.csv", "diabetes.csv"]
query_types = ['count', 'sum', 'mean', 'variance']
privacy_levels = ["very_high", "high", "moderate", "low", "very_low"]
num_iterations = 100

# Adjusting figure size to prevent overlapping
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

for query_type_idx, query_type in enumerate(query_types):
    y_axis_limits = []
    for dataset_path in dataset_paths:
        average_errors = []
        df = pd.read_csv(dataset_path)
        data = df['Age']

        for level in privacy_levels:
            # ... [Your existing DP and error calculation logic here]
            total_error = 0
            for _ in range(num_iterations):
                # Run your DP algorithm and calculate error
                privacy_library = DifferentialPrivacyLibrary(dataset_path, "Age", query_type, level)
                resulting_epsilon = privacy_library.run_differential_privacy()

                # Calculate true result
                true_result = np.sum(data) if query_type == 'sum' else \
                              len(data) if query_type == 'count' else \
                              np.mean(data) if query_type == 'mean' else \
                              statistics.variance(data)

                # Calculate sensitivity, noise, and dp_result
                # ... [Insert your sensitivity and noise calculation logic here]
                ## true results
                #df  = pd.read_csv("csv_dataset.csv")
                #data = df['Age']
                true_count = len(data)
                true_sum = np.sum(data)
                true_mean = np.mean(data)
                true_variance = statistics.variance(data)
                # # sensitivity calculation

                if query_type == 'count':
                    sensitivity = 1
                elif query_type == 'sum':
                    sensitivity = np.max(data) - np.min(data)
                elif query_type == 'mean':
                    sensitivity = (np.max(data) - np.min(data)) / len(data)
                elif query_type == 'variance':
                    sensitivity = (((np.max(data) - np.min(data))**2) * (len(data)/(len(data)**2 - 1)))


                # noise calculation

                beta = sensitivity / resulting_epsilon
                noise = np.random.laplace(0, beta, 1)

                if query_type == 'count':
                    dp_result = true_count + noise
                elif query_type == 'sum':
                    dp_result = true_sum + noise
                elif query_type == 'mean':
                    dp_result = true_mean + noise
                elif query_type == 'variance':
                    dp_result = true_variance + noise
                else:
                    raise ValueError("Invalid query type")
                
                # Calculate error
                error = dp_result - true_result
                total_error += abs(error)

            # Calculate average error
            average_error = total_error / num_iterations
            average_errors.append(average_error)

        y_axis_limits.append((min(average_errors), max(average_errors)))

        # Plotting the average errors for each dataset
        axs[query_type_idx].plot(privacy_levels, average_errors, marker='o', label=f'len(data)={len(data)}')
        axs[query_type_idx].set_title(f"{query_type.capitalize()} Query", size=16)
        axs[query_type_idx].grid(True)
        axs[query_type_idx].tick_params(axis='x', labelsize=14)
        axs[0].tick_params(axis='y', labelsize=14)  
        axs[1].tick_params(axis='y', labelsize=14)  
        axs[2].tick_params(axis='y', labelsize=14)  
        axs[3].tick_params(axis='y', labelsize=14) 

        # Adding legend only to the first subplot
        if query_type_idx == 0:
            axs[query_type_idx].legend(prop={'size': 13})

    # Set individual y-axis limits
    axs[query_type_idx].set_ylim(min(y_axis_limits)[0], max(y_axis_limits)[1])
        # Adjust x-ticks: Show only for bottom subplots (indices 2 and 3)
    if query_type_idx < 2:
        axs[query_type_idx].tick_params(axis='x', labelsize=14, labelbottom=False)
    else:
        axs[query_type_idx].tick_params(axis='x', labelsize=14, labelbottom=True)

# Adjust position of the global x and y labels
#fig.text(0.5, 0.01, 'Privacy Levels', ha='center', va='center', size=16)
fig.text(0.015, 0.5, 'Average Error', va='center', rotation='vertical', size=16)

plt.tight_layout(pad=2.5)
plt.show()

