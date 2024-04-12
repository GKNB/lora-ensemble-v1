# This script will be used to get each of the PPI data from datasets 4 and 5
# Dataset 4+5 ref: https://github.com/Fengithub/symLMF-PPI/tree/master/datasets/neurodegenerative_disease

import random

class d4_5_processor:

    def __init__(self, data_file_path_1, data_file_path_2):
        self.data_file_path_1, self.data_file_path_2 =  data_file_path_1, data_file_path_2
        random.seed(42)

    def load_data(self):
        
        # Save information from the protein index for fast lookup
        word_pairs = {}
        with open(self.data_file_path_2, 'r') as file:
            for line in file:
                words = line.strip().split('\t')
                # Ensure there are at least 6 words in the line to avoid IndexError
                if len(words) > 5:
                    key, value = words[1], words[5]
                    word_pairs[key] = value

        pos_pairs = []
        neg_pairs = []

        # Process the second file for interactions
        with open(self.data_file_path_1, 'r') as file:
            for line in file:
                words = line.strip().split('\t')
                # Ensure the line has at least 3 words for protein1, protein2, and interaction type
                if len(words) > 2:
                    prot1, prot2, interaction_type = words[0], words[1], words[2]

                    # Look up the protein names using the dictionary
                    protein1_name = word_pairs.get(prot1, "Unknown")
                    protein2_name = word_pairs.get(prot2, "Unknown")

                    # Depending on the interaction type, add to the respective list
                    if interaction_type == '1':
                        pos_pairs.append((protein1_name, protein2_name))
                    elif interaction_type == '0':
                        neg_pairs.append((protein1_name, protein2_name))

        # Returns 2 lists, positive protein interactions and negative interactions
        return pos_pairs, neg_pairs