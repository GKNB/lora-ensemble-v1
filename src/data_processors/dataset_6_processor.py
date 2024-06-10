# This script will be used to get each of the PPI data from dataset 6
# Dataset 6 ref: https://pubmed.ncbi.nlm.nih.gov/34727106/

import random

class d6_processor:

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        random.seed(42)


    def load_data(self):
        positive_interactions = []
        negative_interactions = []
        
        try:
            with open(self.data_file_path, 'r') as file:
                # Skipping the header line
                next(file)

                # Reading the file without a specified number of lines limit
                for line in file:
                    parts = line.strip().split()
                    if parts:  # checking if the line is not empty
                        # Creating a list with proteinA, proteinB, and interaction ("yes" or "no")
                        interaction = "yes" if parts[2] == 'True' or parts[2] == '1' else "no"
                        if interaction == "yes":
                            positive_interactions.append((parts[0], parts[1]))
                        else:
                            negative_interactions.append((parts[0], parts[1]))

        except FileNotFoundError as e:
            print(f"The file does not exist. Error: {str(e)}")

        # Ensuring negative_interactions list is the same length as positive_interactions list
        if len(negative_interactions) > len(positive_interactions):
            negative_interactions = random.sample(negative_interactions, len(positive_interactions))

        # After filling positive_interactions and negative_interactions
        overlapping_interactions = set(positive_interactions) & set(negative_interactions)
        if overlapping_interactions:
            raise ValueError(f"Overlapping interactions found: {overlapping_interactions}")

        # Returning the lists of positive and negative interactions
        return positive_interactions, negative_interactions
