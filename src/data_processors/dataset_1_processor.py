# This script will be used to get each of the 3 radiation groups data from dataset 1
# Dataset 1 ref: https://www.storedb.org/store_v3/study.jsp?studyId=1153

import random
import pandas as pd

class d1_processor:

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        random.seed(42)

    def load_excel_spreadsheets(self):
        spreadsheets = [
            ('S6', 'S3'),
            ('S7', 'S4'),
            ('S8', 'S5'),
        ]

        proteins_063gy = []
        proteins_0125 = []
        proteins_05 = []
        datasets = [
            proteins_063gy,
            proteins_0125,
            proteins_05,
        ]

        for i, sheet in enumerate(spreadsheets):

            # Load data from sheet at first index
            df = pd.read_excel(self.data_file_path, sheet_name=sheet[0])
            list1 = []
            name_list = df['Unnamed: 3'].tolist()[2:]
            info_list = df['Unnamed: 4'].tolist()[2:]
            for name, info in zip(name_list, info_list):
                info = info.strip()
                name_info_dict = {name: info}
                list1.append(name_info_dict)

            # Filter out dictionaries with duplicate keys, keeping the first occurrence of each key
            seen_names = set()
            unique_list1 = []
            for item in list1:
                name = next(iter(item))
                if name not in seen_names:
                    unique_list1.append(item)
                    seen_names.add(name)
            list1 = unique_list1

            # Load data from sheet at the second index
            df = pd.read_excel(self.data_file_path, sheet_name=sheet[1])
            list2 = []
            new_data = df['Unnamed: 5'].tolist()
            new_data = new_data[2:]

            # Extract the name before "|" and strip spaces
            for item in new_data:
                # Extract name before "|" and strip spaces
                name = item.split('|')[0].strip()  

                # Split the string by "|" and extract the info after the second "|"
                parts = item.split('|')
                info = parts[2] 

                # Now find the first "[" in the info
                bracket_pos = info.find('[')
                if bracket_pos != -1:  # If a "[" is found
                    info = info[:bracket_pos]  # Extract info up to the "["

                info = info.strip()  # Strip any leading/trailing whitespace

                # create dictionary element and append to list
                name_info_dict = {name: info}
                list2.append(name_info_dict)
            
            # Extract names from dictionaries in list1 for checking
            names_list1 = {list(item.keys())[0] for item in list1}
            list3 = []
            while len(list3) < len(list1):
                random_dict = random.choice(list2)  # Select a random dictionary from list2
                random_name = list(random_dict.keys())[0]  # Extract the name (key) from the randomly selected dictionary
                
                # Check if the random name is not among the names in list1 and not already in list3 based on names
                if random_name not in names_list1 and all(random_name != list(d.keys())[0] for d in list3):
                    list3.append(random_dict)

            # Finally, convert the dictionaries into the correct naming convention
            deregulated_proteins = []
            for item in list1:
                name, info = list(item.items())[0]
                formatted_name = f"{name} ({info})"
                deregulated_proteins.append(formatted_name)

            unaffected_proteins = []
            for item in list3:
                name, info = list(item.items())[0]
                formatted_name = f"{name} ({info})"
                unaffected_proteins.append(formatted_name)

            datasets[i] = list(zip(deregulated_proteins, unaffected_proteins))

        # print("Test: ", datasets[0][0])
        # print("Test: ", datasets[0][1])
        # print("Test: ", datasets[0][2])
        # print("Test: ", datasets[1][0])
        
        # For each dataset, index 0 represents a protein name-description pair that is deregulated by LDR exposure
        # For each dataset, index 1 represents a protein name-description pair that is NOT deregulated by LDR exposure
        # The negative samples were taken from the corresponding list of proteins identified
        # they were sampled only if there were not already in the list of proteins affected, thus there are no repeats
        return datasets[0], datasets[1], datasets[2]
        # Datasets 1.1, 1.2, and 1.3
    

    