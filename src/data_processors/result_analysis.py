# This script is used to analyze the results of each experiment
# It makes 2 lists, correctly identified proteins and incorrectly identified proteins
# for datasets 4-6, it lists the correctly and incorrectly identified pairs of proteins

import re

class result_analysis:
    def init(self, file_name):
        # Initialize with the file name
        self.file_name = file_name
        self.file_path = f"../results/experiments_v1/{file_name}"

    def load_data(self):
        # Open and read the text file
        with open(self.file_path  + ".txt", 'r', encoding='utf-8') as file:
            return file.read()

    def run(self):
        # Load the data from the file
        text = self.load_data()

        # Extracting information 

        # Use for dataset 1
        # prompts = re.findall(r'Prompt \d+: .*? of (.*?) \(.*?\)\s.*?True Label: (\w+)', text, re.DOTALL)
        # predictions = re.findall(r'Model Prediction \d+: (\w+)', text, re.DOTALL)

        # Use for dataset 2
        # prompts = re.findall(r'Prompt \d+: Given the options yes or no, will there be significant deregulation of the protein (.*?) .*?exposure to low dose radiation at 2\.0 Gy\?\nTrue Label: (\w+)', text, re.DOTALL)
        # predictions = re.findall(r'Model Prediction \d+: (\w+)', text, re.DOTALL)

        # Use for dataset 3
        # prompts = re.findall(r'Prompt \d+: Given the options yes or no, will there be an altered acetylation status of protein (.*?) after exposure to low dose radiation at 0\.5 Gy\?\nTrue Label: (\w+)', text, re.DOTALL)
        # predictions = re.findall(r'Model Prediction \d+: (\w+)', text, re.DOTALL)

        # Use for dataset 3c
        # prompts = re.findall(r'Prompt \d+: Given the options yes or no, will there be deregulation of the protein (.*?) after low-dose radiation exposure\?\nTrue Label: (\w+)', text, re.DOTALL)
        # predictions = re.findall(r'Model Prediction \d+: (\w+)', text, re.DOTALL)

        # Lists to store correct and incorrect protein names
        # correct_proteins = []
        # incorrect_proteins = []

        # # Comparing true labels and predictions
        # for idx, (protein, true_label) in enumerate(prompts): 
        #     if idx < len(predictions):
        #         is_correct = (true_label.lower() == predictions[idx].lower())
        #         if is_correct:
        #             correct_proteins.append(protein.split()[0])
        #         else:
        #             incorrect_proteins.append(protein.split()[0])

        # # Output sorted results
        # if correct_proteins:
        #     print("Correct:")
        #     for protein in correct_proteins:
        #         print(protein.split()[0])
        
        # if incorrect_proteins:
        #     print("\nIncorrect:")
        #     for protein in incorrect_proteins:
        #         print(protein.split()[0])

        # # Print counts
        # print(f"\nCorrect predictions: {len(correct_proteins)}")
        # print(f"Incorrect predictions: {len(incorrect_proteins)}")

        # # Write results to a file
        # with open(f"../results/analysis/{self.file_name}_analysis.txt", 'w', encoding='utf-8') as file:
        #     if correct_proteins:
        #         file.write("Correct:\n")
        #         for protein in correct_proteins:
        #             file.write(f"{protein}\n")
            
        #     if incorrect_proteins:
        #         file.write("\nIncorrect:\n")
        #         for protein in incorrect_proteins:
        #             file.write(f"{protein}\n") 

        #     # Write counts
        #     file.write(f"\nCorrect predictions: {len(correct_proteins)}\n")
        #     file.write(f"Incorrect predictions: {len(incorrect_proteins)}\n")



        # Use for datasets 4, 5, and 6
        prompts = re.findall(r'Prompt \d+: Given the options yes or no, is there a protein interaction between (.*?) and (.*?) in the presence of (.*?)\?\nTrue Label: (\w+)', text, re.DOTALL)
        predictions = re.findall(r'Model Prediction \d+: (\w+)', text, re.DOTALL)

        # Lists to store correct and incorrect protein pairs
        correct_proteins = []
        incorrect_proteins = []

        # Comparing true labels and predictions
        for idx, (protein1, protein2, disease, true_label) in enumerate(prompts):
            if idx < len(predictions):
                is_correct = (true_label.lower() == predictions[idx].lower())
                protein_pair = f"{protein1}, {protein2}" 
                if is_correct:
                    correct_proteins.append(protein_pair)
                else:
                    incorrect_proteins.append(protein_pair)

        # Output sorted results
        if correct_proteins:
            print("Correct:")
            for protein in correct_proteins:
                print(protein)
        
        if incorrect_proteins:
            print("\nIncorrect:")
            for protein in incorrect_proteins:
                print(protein)

        # Print counts
        print(f"\nCorrect predictions: {len(correct_proteins)}")
        print(f"Incorrect predictions: {len(incorrect_proteins)}")

        # Output sorted results and write results to a file
        with open(f"../results/analysis/{self.file_name}_analysis.txt", 'w', encoding='utf-8') as file:
            if correct_proteins:
                file.write("Correct:\n")
                for protein_pair in correct_proteins:
                    file.write(f"{protein_pair}\n")
            
            if incorrect_proteins:
                file.write("\nIncorrect:\n")
                for protein_pair in incorrect_proteins:
                    file.write(f"{protein_pair}\n")

            # Write counts
            file.write(f"\nCorrect predictions: {len(correct_proteins)}\n")
            file.write(f"Incorrect predictions: {len(incorrect_proteins)}\n")
