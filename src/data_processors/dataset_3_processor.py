# This script will be used to get each of the 2 radiation groups data from dataset 3
# Datset 3 ref: https://www.storedb.org/store_v3/study.jsp?studyId=1039

import random
import pandas as pd

class d3_processor:

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        random.seed(42)

    def load_excel_spreadsheets(self):

        spreadsheets = ['S4', 'S5']
        datasets = [] 

        for sheet in spreadsheets:
            # Load data from the specified sheet
            df = pd.read_excel(self.data_file_path, sheet_name=sheet)

            # List1: Unique protein names from the first column starting at index 3
            list1 = list(df.iloc[3:, 0])

            # Load another dataset for randomly sampling proteins (from table S1, column 11)
            df_s1 = pd.read_excel(self.data_file_path, sheet_name='S1')
            
            # Extract protein names before the first "|", starting at index 3
            all_proteins = [name.split('|')[0].strip() for name in df_s1.iloc[3:, 10] if isinstance(name, str)]

            # Removing duplicates from all_proteins before sampling
            unique_proteins = []
            [unique_proteins.append(x) for x in all_proteins if x not in unique_proteins]
            all_proteins = unique_proteins

            # Prepare to sample for list2, exclude proteins from list1
            available_proteins_for_list2 = [protein for protein in all_proteins if protein not in list1]

            # Reset the random seed right before sampling to ensure determinism
            random.seed(42)

            # Randomly sample proteins, ensuring no overlap with list1
            list2 = random.sample(available_proteins_for_list2, min(len(list1), len(available_proteins_for_list2)))

            # Add the processed data to the datasets list
            datasets.append((list1, list2))

        # print("datasets: ", datasets[0][0])
        # print("datasets: ", datasets[0][1])

        return datasets
                
