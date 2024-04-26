# This script is used to facilitate loading the data into json files to be used later

import json

class dataset_loader:

    def __init__(self):
        pass

    def load_datasets(self):

        # Uncomment the code below for the dataset and prompt being used
        # This is only necessary for changing the prompts and saving them as json files
        # For loading the saved json files skip to the next function
        self.list1 = []
        self.list2 = []

        # Dataset 1
        # set 1: 446 x 2 = 892
        # set 2: 666 x 2 = 1,332
        # set 3: 102 x 2 = 204
        # prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.063 Gy?" 
        # prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.125 Gy?"
        # prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.5 Gy?"
        from dataset_1_processor import d1_processor
        data_file_path = "/direct/sdcc+u/rengel/data/dataset_1_original.xls"
        p1 = d1_processor(data_file_path)
        d1, d2, d3 = p1.load_excel_spreadsheets() 
        for item in d3: # Change this to d1, d2, or d3 depending on the subset of data
            self.list1.append(item[0])  
            self.list2.append(item[1]) 
        print("Length of dataset: ", len(self.list1))
        print("Length of dataset: ", len(self.list2))


        # Dataset 2
        # set 1: 80 x 2 = 160
        # set 2: 99 x 2 = 198
        # set 3: 37 x 2 = 74
        # set 4: 47 x 2 = 94
        # prompt = f"Given the options yes or no, will there be significant deregulation of the protein {protein} 72 hours after exposure to low dose radiation at 2.0 Gy?"
        # prompt = f"Given the options yes or no, will there be significant deregulation of the protein {protein} 1 month after exposure to low dose radiation at 2.0 Gy?"
        # prompt = f"Given the options yes or no, will there be significant deregulation of the protein {protein} 3 months after exposure to low dose radiation at 2.0 Gy?"  
        # prompt = f"Given the options yes or no, will there be significant deregulation of the protein {protein} 6 months after exposure to low dose radiation at 2.0 Gy?"
        # from dataset_2_processor import d2_processor
        # data_file_path = "/direct/sdcc+u/rengel/data/dataset_2_original.xlsx"
        # p2 = d2_processor(data_file_path)
        # d1, d2, d3, d4 = p2.load_excel_spreadsheets()
        # for item in d4[0]:
        #     self.list1.append(item)  
        # for item in d4[1]:
        #     self.list2.append(item)  
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))
        

        # Dataset 3
        # set 1: 49 x 2 = 98
        # set 2: 77 x 2 = 154
        # prompt = f"Given the options yes or no, will there be an altered acetylation status of protein {protein} 4 hours after exposure to low dose radiation at 0.5 Gy?" 
        # prompt = f"Given the options yes or no, will there be an altered acetylation status of protein {protein} 24 hours after exposure to low dose radiation at 0.5 Gy?" 
        # from dataset_3_processor import d3_processor
        # data_file_path = "/direct/sdcc+u/rengel/data/dataset_3_original.xlsx"
        # p3 = d3_processor(data_file_path)
        # d1, d2  = p3.load_excel_spreadsheets()
        # for item in d1[0]:
        #     self.list1.append(item)  
        # for item in d1[1]:
        #     self.list2.append(item)  
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))


        # Dataset 3c
        # 1,111 x 2 = 2,222 
        # prompt = f"Given the options yes or no, will there be deregulation of the protein {protein} after low-dose radiation exposure?"
        # from dataset_3c_processor import d3c_processor
        # data_file_path_1 = "/direct/sdcc+u/rengel/data/dataset_1_original.xls"
        # data_file_path_2 = "/direct/sdcc+u/rengel/data/dataset_2_original.xlsx"
        # data_file_path_3 = "/direct/sdcc+u/rengel/data/dataset_3_original.xlsx"
        # p3c = d3c_processor(data_file_path_1, data_file_path_2, data_file_path_3)
        # d1, d2 = p3c.load_excel_spreadsheets()
        # for item in d1:
        #     self.list1.append(item)
        # for item in d2:
        #     self.list2.append(item)
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))


        # Dataset 4
        # 5,881 x 2 = 11,762 pairs
        # prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of neurodegenerative diseases?"
        # from dataset_4_5_processor import d4_5_processor
        # data_file_path_1 = "/direct/sdcc+u/rengel/data/dataset_4_original_pros.txt"
        # data_file_path_2 = "/direct/sdcc+u/rengel/data/dataset_4_original_index.txt"
        # p4 = d4_5_processor(data_file_path_1, data_file_path_2)
        # d1, d2  = p4.load_data()
        # for item in d1: self.list1.append(item)   
        # for item in d2: self.list2.append(item) 
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))
        

        # Dataset 5
        # 5,131 x 2 = 10,262 pairs
        # prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of metabolic diseases?"
        # from dataset_4_5_processor import d4_5_processor
        # data_file_path_1 = "/direct/sdcc+u/rengel/data/dataset_5_original_pros.txt"
        # data_file_path_2 = "/direct/sdcc+u/rengel/data/dataset_5_original_index.txt"
        # p5 = d4_5_processor(data_file_path_1, data_file_path_2)
        # d1, d2  = p5.load_data()
        # for item in d1: self.list1.append(item)   
        # for item in d2: self.list2.append(item)
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))

        # Dataset 6
        # 933 x 2 = 1,866 pairs
        # prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of cancer?"
        # from dataset_6_processor import d6_processor
        # data_file_path = "/direct/sdcc+u/rengel/data/dataset_6_original.txt"
        # p4 = d6_processor(data_file_path)
        # d1, d2  = p4.load_data()
        # for item in d1: self.list1.append(item)   
        # for item in d2: self.list2.append(item)
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))



        # The following code is used to save the datasets/prompts into json files
       
        # Datasets 1-3
        # Copy/paste the prompt from above to use for each list
        self.dataset_examples = []
        for protein in self.list1:
            prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.5 Gy?" 
            self.dataset_examples.append({'question': prompt, 'answer': 'Yes'})
        for protein in self.list2:
            prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.5 Gy?" 
            self.dataset_examples.append({'question': prompt, 'answer': 'No'})

        # Datasets 4-6
        # Copy/paste the prompt from above to use for each list
        # self.dataset_examples = [] 
        # for pos_pair in self.list1:
        #     protein1 = pos_pair[0]
        #     protein2 = pos_pair[1]
        #     prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of cancer?"
        #     self.dataset_examples.append({'question': prompt, 'answer': 'Yes'})
        # for neg_pair in self.list2:
        #     protein1 = neg_pair[0]
        #     protein2 = neg_pair[1]
        #     prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of cancer?"
        #     self.dataset_examples.append({'question': prompt, 'answer': 'No'})

        # Use this code to save the prompts to json files
        with open('/direct/sdcc+u/rengel/data/dataset_1_v3_prompts.json', 'w') as file:
            json.dump(self.dataset_examples, file, indent=4)
