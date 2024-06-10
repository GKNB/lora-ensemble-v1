# This script will be used to get each of the 4 radiation groups data from dataset 2
# Datset 2 ref: https://www.storedb.org/store_v3/study.jsp?studyId=1107

import random
import pandas as pd

class d2_processor:

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        random.seed(42)

    def load_excel_spreadsheets(self):
        datasets = []

        # Define the columns for fold-change values for each dataset iteration
        fold_change_columns = [4, 6, 8, 10] 

        for col in fold_change_columns:
            # Load data from the Excel file
            df = pd.read_excel(self.data_file_path)

            # Convert both the fold_change column and the p-value column to float
            df.iloc[:, col - 1] = pd.to_numeric(df.iloc[:, col - 1], errors='coerce')  # Convert fold-change to float
            df.iloc[:, col] = pd.to_numeric(df.iloc[:, col], errors='coerce')  # Convert p-value to float

            list1 = []
            list2 = []

            for index, row in df.iterrows():
                if index >= 2:  # Skip to the third row (indexing starts from 0)
                    protein_name = row.iloc[1]  # Protein name from column 2
                    fold_change = row.iloc[col - 1]  # Fold-change from the specified column
                    p_value = row.iloc[col]  # P-value from the column next to fold-change

                    if pd.notnull(fold_change) and pd.notnull(p_value):
                        list1.append((protein_name, fold_change, p_value))
                        
                        if ((fold_change <= 0.77 or fold_change >= 1.3) and p_value <= 0.05):
                            list2.append(protein_name)

            # Filter list1 to only include proteins not in list2
            candidates_for_list3 = [name for name, _, _ in list1 if name not in list2]
            
            # Randomly sample from candidates to match the size of list2
            list3 = random.sample(candidates_for_list3, min(len(list2), len(candidates_for_list3)))
            assert len(list2) == len(list3), "Lists do not match sizes"

            # List 2 is the proteins deregulated, list 3 is proteins unaffected
            datasets.append((list2, list3))

        return datasets