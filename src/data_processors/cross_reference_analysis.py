# This script is used to conduct a cross-reference analysis of datasets 3c, 4, 5, and 6
# It first loads the datasets from their respective original data files
# Then it extracts the protein names for the proteins that are affected by LDR from dataset 3c
# It then extracts the protein interactions that are positive for datasets 4, 5, and 6
# Finally, we check how many of the interactions involve proteins from 3c

import re
import json
from collections import Counter

class cross_reference_analysis:

    def run(self):

        # Get positive proteins from dataset 3c:
        with open('/direct/sdcc+u/rengel/data/dataset_3c_prompts.json', 'r') as file:
            prompts_data = json.load(file)

        ldr_proteins = []
        for item in prompts_data:
            if item['answer'].lower() == "yes": 
                match = re.search(r'\b(\w+)\b(?= after)', item['question'])
                if match:
                    ldr_proteins.append(match.group(1))

        # Get positive interactions from either dataset 4, dataset 5, or dataset 6
        with open('/direct/sdcc+u/rengel/data/dataset_4_prompts.json', 'r') as file:
            interaction_data = json.load(file) 

        # DEBUG
        # print("Number of Interaction Pairs: ", len(interaction_data))

        interaction_proteins = []
        interaction_proteins_counter = Counter()
        neg_proteins = []
        other = []
        for item in interaction_data:
            if item['answer'].lower() == "yes":
                matches = re.findall(r'between ((?:[\w-]+(?:; ?[\w-]+)*)+) and ([\w-]+)', item['question'])
                if matches:
                    for match in matches:
                        interaction_proteins.append(match[0].lower())
                        interaction_proteins.append(match[1].lower())
                        interaction_proteins_counter.update([match[0].lower(), match[1].lower()])
                else:
                    other.append((item['answer'], item['question']))
            elif item['answer'].lower() == "no":
                matches = re.findall(r'between ((?:[\w-]+(?:; ?[\w-]+)*)+) and ([\w-]+)', item['question'])
                if matches:
                    for match in matches:
                        neg_proteins.append(match[0].lower())
                        neg_proteins.append(match[1].lower())
                else:
                    other.append((item['answer'], item['question']))
            else:
                print(item['answer'].lower())
                
        # DEBUG
        # print("Number of Positive Proteins: ", len(interaction_proteins))
        # print("Number of Negative Proteins: ", len(neg_proteins))
        # print("other: ", other)

        # Convert to sets for unique datasets
        unique_ldr_proteins = set(ldr_proteins)
        unique_interaction_proteins = set(interaction_proteins)

        # Analyze the overlap between both sets
        overlapping_proteins = []
        for protein in interaction_proteins:
            if protein in ldr_proteins:
                overlapping_proteins.append(protein)
        overlapping_proteins = overlapping_proteins
        unique_overlapping_proteins = set(overlapping_proteins)

        # Calculate Unique Percentage and Jaccard Index
        percentage = (len(unique_overlapping_proteins) / len(unique_interaction_proteins))
        jaccard_index = len(unique_overlapping_proteins) / (len(unique_ldr_proteins) + len(unique_interaction_proteins) - len(unique_overlapping_proteins))

        # Multiset Coverage
        total_interactions = len(interaction_proteins)
        multiset_coverage = len(overlapping_proteins) / total_interactions if total_interactions else 0

        # Weighted Jaccard Index
        weighted_jaccard_index = len(overlapping_proteins) / (len(interaction_proteins) + len(ldr_proteins) - len(overlapping_proteins))

        # Print all protein names that are in both datasets
        print("Unique Overlapping Proteins: ", unique_overlapping_proteins)
        print(f"\nPercentage of Overlap (Unique): {percentage*100:.2f}%")
        print(f"Jaccard Index (Unique): {jaccard_index:.4f}")
        print(f"Multiset Coverage: {multiset_coverage*100:.2f}%")
        print(f"Weighted Jaccard Index: {weighted_jaccard_index:.4f}")
        print(f"\nNumber of Overlapping Proteins: {len(overlapping_proteins)}")
        print(f"Number of LDR Proteins: {len(ldr_proteins)}")
        print(f"Number of Interacting Proteins: {len(interaction_proteins)}")
        print(f"\nNumber of Unique Overlapping Proteins: {len(unique_overlapping_proteins)}")
        print(f"Number of Unique LDR-affected Proteins: {len(ldr_proteins)}")
        print(f"Number of Unique Interacting Proteins: {len(unique_interaction_proteins)}")

        # Write results to a file, change file name depending on the dataset used
        with open('/direct/sdcc+u/rengel/results/analysis/cross_reference_analysis/neurodegenerative.txt', 'w') as file: 
            file.write("Unique Overlapping Proteins: {}\n".format(unique_overlapping_proteins))
            file.write("\nPercentage of Overlap (Unique): {:.2f}%\n".format(percentage*100))
            file.write("Jaccard Index (Unique): {:.4f}\n".format(jaccard_index))
            file.write("Multiset Coverage: {:.2f}%\n".format(multiset_coverage*100))
            file.write("Weighted Jaccard Index: {:.4f}\n".format(weighted_jaccard_index))
            file.write("\nNumber of Overlapping Proteins: {}\n".format(len(overlapping_proteins)))
            file.write("Number of LDR Proteins: {}\n".format(len(ldr_proteins)))
            file.write("Number of Interacting Proteins: {}\n".format(len(interaction_proteins)))
            file.write("\nNumber of Unique Overlapping Proteins: {}\n".format(len(unique_overlapping_proteins)))
            file.write("Number of Unique LDR-affected Proteins: {}\n".format(len(ldr_proteins)))
            file.write("Number of Unique Interacting Proteins: {}\n".format(len(unique_interaction_proteins)))    
