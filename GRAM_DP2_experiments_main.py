import numpy as np
import pandas as pd

class DifferentialPrivacyLibrary:
    def __init__(self, dataset_path, column_name, query_type, privacy_level):
        self.dataset_path = dataset_path
        self.column_name = column_name
        self.query_type = query_type
        self.privacy_level = privacy_level
        self.privacy_level_ranges = {
            'very_high': (0.01, 0.10),
            'high': (0.10, 0.25),
            'moderate': (0.25, 0.50),
            'low': (0.50, 0.75),
            'very_low': (0.75, 1.0)
        }

    def initialize_epsilon(self):
        epsilon_range = self.privacy_level_ranges.get(self.privacy_level)
        if epsilon_range is None:
            raise ValueError("Invalid privacy level")

        initial_epsilon = np.random.uniform(*epsilon_range)
        return initial_epsilon

    def get_data_from_column(self):
        # Assuming the dataset is in CSV format, adjust accordingly for other formats
        data = pd.read_csv(self.dataset_path)[self.column_name]
        return data

    def calculate_sensitivity(self):
        data = self.get_data_from_column()
        # Implement sensitivity calculation based on the query type
        if self.query_type == 'count':
            sensitivity = 1
        elif self.query_type == 'sum':
            # Implement sensitivity calculation for sum
            sensitivity = np.max(data) - np.min(data)
        elif self.query_type == 'mean':
            # Implement sensitivity calculation for mean
            sensitivity = (np.max(data) - np.min(data)) / len(data)
        elif self.query_type == 'variance':
            # Implement sensitivity calculation for variance
            # correct it: ((M-m)^2) * (n/(n^2-1))
            sensitivity = (((np.max(data) - np.min(data))**2) * (len(data)/(len(data)**2 - 1)))
            #sensitivity = ((np.max(data) - np.min(data)) / len(data)) ** 2 * (len(data) - 1)
        else:
            raise ValueError("Invalid query type")

        return sensitivity

    def calculate_beta(self, sensitivity, initial_epsilon):
        beta = sensitivity / initial_epsilon
        return beta

    def calculate_epsilon(self, sensitivity, laplace, risk_threshold, impact_of_disclosure):
        #epsilon = (sensitivity / laplace) * (1 - (impact_of_disclosure / risk_threshold))
        epsilon = (sensitivity / laplace) * ((risk_threshold - impact_of_disclosure) / risk_threshold)
        return epsilon

    def initialize_risk_threshold(self):
        # Assuming an inverse relationship between risk threshold and desired privacy
        # You may need to adjust the specifics based on your requirements
        privacy_levels = list(self.privacy_level_ranges.keys())
        risk_thresholds = [1.0 / (i + 1) for i in range(len(privacy_levels))]

        if self.privacy_level not in privacy_levels:
            raise ValueError("Invalid privacy level")

        index = privacy_levels.index(self.privacy_level)
        risk_threshold = risk_thresholds[index]

        return risk_threshold

    def assess_dependence(self):
        print("Qualitative Assessment of Importance and Sensitivity of Inserted Dataset")
        print("-----------------------------------------------------------------------")

        # Prompt for user input on various factors
        data_relevance = 5
        data_sensitivity = 5
        data_uniqueness = 5
        data_volume = 5
        data_source_trust = 5

        # Maximum possible score for each question
        max_score_per_question = 5

        # Calculate a weighted qualitative score on a scale from 0 to 5
        total_score = (
            (data_relevance / max_score_per_question) +
            (data_sensitivity / max_score_per_question) +
            (data_uniqueness / max_score_per_question) +
            (data_volume / max_score_per_question) +
            (data_source_trust / max_score_per_question)
        )

        # Map the total score to the qualitative categories
        if total_score > 4:
            qualitative_score = 'High'
        elif 3 <= total_score <= 4:
            qualitative_score = 'Moderate'
        elif 2 <= total_score < 3:
            qualitative_score = 'Low'
        elif 1 <= total_score < 2:
            qualitative_score = 'Very Low'
        else:
            qualitative_score = 'Extremely Low'

        return qualitative_score

    def calculate_factor_1_score(self):
        # Get the qualitative score from the user
        qualitative_score = self.assess_dependence()

        # Factor 1 weights for each category
        factor_1_weights = {
            'High': 1.0,
            'Moderate': 0.75,
            'Low': 0.5,
            'Very Low': 0.25,
            'Extremely Low': 0.1
        }

        return factor_1_weights.get(qualitative_score, 0)

    def calculate_factor_2_score(self, column_name):
        # Factor 2 PII Identifiers and their percentages
        factor_2_weights = {
            'Name': 0.29,
            'Age': 0.10,
            'SSN': 0.20,
            'Address': 0.12,
            'DOB': 0.12,
            'Credit_Card_Information': 0.07,
            'Phone': 0.06,
            'Email': 0.05,
            'Medical_Records': 0.02
        }

        return factor_2_weights.get(column_name, 0)

    def run_differential_privacy(self):
        # Factor 1
        factor_1_score = self.calculate_factor_1_score()

        # Factor 2
        factor_2_score = self.calculate_factor_2_score(self.column_name)

        # Calculate IDD as the average of the two factors
        IDD = (factor_1_score + factor_2_score) / 2

        # Remaining logic for differential privacy calculation (unchanged from the original code)
        initial_epsilon = self.initialize_epsilon()
        sensitivity = self.calculate_sensitivity()
        beta = self.calculate_beta(sensitivity, initial_epsilon)
        laplace_noise = np.random.laplace(0, beta)
        risk_threshold = self.initialize_risk_threshold()

        epsilon = self.calculate_epsilon(sensitivity, laplace_noise, risk_threshold, IDD)
        return abs(epsilon)

# # Example usage
# dataset_path = "csv_dataset.csv"
# column_name = "Age"
# query_type = "sum"  # Replace with the desired query type
# privacy_level = input("Select privacy level (very_high, high, moderate, low, very_low): ")  # Prompt the user for privacy level

# privacy_library = DifferentialPrivacyLibrary(dataset_path, column_name, query_type, privacy_level)
# resulting_epsilon = privacy_library.run_differential_privacy()
# print(f"Resulting Epsilon: {resulting_epsilon}")
