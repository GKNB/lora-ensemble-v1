# This script will be used to get each of the 3 radiation groups data from dataset 1
# Dataset 1 ref: https://www.storedb.org/store_v3/study.jsp?studyId=1153

import random
import pandas as pd

class d3c_processor:

    def __init__(self, data_file_path_1, data_file_path_2, data_file_path_3):
        self.data_file_path_1 = data_file_path_1
        self.data_file_path_2 = data_file_path_2
        self.data_file_path_3 = data_file_path_3
        random.seed(42)

    def load_excel_spreadsheets(self):
        
        # Call dataset 1 processor, and combine all the elements into list1
        from dataset_1_processor import d1_processor
        p1 = d1_processor(self.data_file_path_1)
        d1, d2, d3 = p1.load_excel_spreadsheets()
        list1 = []
        list2 = []
        # For each item in each of the 3 subsets, the first element is the affected protein, the second element is the unaffected protein
        for item in d1:
            protein = item[0].strip().split()[0].lower()
            list1.append(protein)  
            protein = item[1].strip().split()[0].lower()
            list2.append(protein) 

        for item in d2:
            protein = item[0].strip().split()[0].lower()
            list1.append(protein)  
            protein = item[1].strip().split()[0].lower()
            list2.append(protein) 

        for item in d3:
            protein = item[0].strip().split()[0].lower()
            list1.append(protein)  
            protein = item[1].strip().split()[0].lower()
            list2.append(protein) 


        # Call dataset 2 processor, and combine all the elements into list2
        from dataset_2_processor import d2_processor
        p2 = d2_processor(self.data_file_path_2)
        d1, d2, d3, d4 = p2.load_excel_spreadsheets()
        datasets = [d1, d2, d3, d4]
        for dataset in datasets:
            for item in dataset[0]:
                protein = item.strip().lower()
                list1.append(protein)  
            for item in dataset[1]:
                protein = item.strip().lower()
                list2.append(protein)  


        # Call dataset 3 processor, and combine all the elements into list3
        from dataset_3_processor import d3_processor
        p3 = d3_processor(self.data_file_path_3)
        d1, d2  = p3.load_excel_spreadsheets()
        for item in d1[0]:
            protein = item.strip().lower()
            list1.append(protein)
        for item in d1[1]:
            protein = item.strip().lower()
            list2.append(protein)


        # Eliminate duplicates from each individual list
        unique_proteins = []
        [unique_proteins.append(x) for x in list1 if x not in unique_proteins]
        list1 = unique_proteins
        unique_proteins = []
        [unique_proteins.append(x) for x in list2 if x not in unique_proteins]
        list2 = unique_proteins

        # Remove duplicates from list2 that exist in list1
        list2 = [x for x in list2 if x not in list1]


        # Now we randomly sample proteins unaffected from dataset 1 until list2 is equal length to list1
        # Load data from sheet at the second index
        df = pd.read_excel(self.data_file_path_1, sheet_name="S4")
        new_data = df['Unnamed: 5'].tolist()
        new_data = new_data[2:]

        # Extract the name before "|" and strip spaces
        while len(list2) < len(list1):
            # Randomly select an item from new_data
            item = random.choice(new_data)
            name = item.split('|')[0].strip().lower()  # Format new protein name

            # Check if the name already exists in list1 or list2, if it does not, add it to list2
            if name not in list1 and name not in list2:
                list2.append(name)
                # Remove the item from new_data to avoid reselection
                new_data.remove(item)

        print("Length of list1: ", len(list1))
        print("Length of list2: ", len(list2))

        # Return both lists
        # List 1 represents the proteins affected by radiation across datasets 1, 2, and 3
        # List 2 represents the proteins not affected by radiation across datasets 1, 2, and 3
        return list1, list2
    

     