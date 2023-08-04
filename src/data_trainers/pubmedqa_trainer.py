#Necessary Imports
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig, PrefixTuningConfig, LoraConfig
from huggingface_hub import login, HfApi
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn.functional as F
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import re

 

class pubmedqa_trainer():

    def init(self, model, log_file):

        print("Initializing PubMedQA Trainer...")

        self.log_file = log_file
        open(self.log_file, 'w').close()

        self.device = "cuda"
        self.model_name_or_path = model
        self.tokenizer_name_or_path = model
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)

        # PROMPT TUNING CONFIG
        # Optimal results with 9 epochs, lr = .0001 and batch size 4
        self.peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, prompt_tuning_init=PromptTuningInit.TEXT, num_virtual_tokens=8, prompt_tuning_init_text="Generate either yes, no, or maybe: ", tokenizer_name_or_path=self.tokenizer_name_or_path)
        self.model = get_peft_model(self.model, self.peft_config)

        # LORA CONFIG
        # Optimal results with 4 epochs, lr = .00001 and batch size = 4
        self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=8, lora_dropout=.01, bias="lora_only")
        self.model = get_peft_model(self.model, self.peft_config)
        
        # PREFIX TUNING CONFIG (Not tested in this project)
        #self.peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=8)
        #self.model = get_peft_model(self.model, self.peft_config)

        # For combined strategy, optimal results with 4 epochs, lr = .00001 and batch size = 4

        self.lr = .00001
        self.num_epochs = 4
        self.batch_size = 4

        # Note that all experiments used a weight decay of 0 and that changing this value did not yield more accurate results
        self.weight_decay = 0


    def log(self, message):
        # This function handles writing logs to the specified file
        try:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            raise Exception(f"Failed to open and write to file: {self.log_file}") from e
    

    def load_dataset(self):

        print("Loading Dataset...")

        self.dataset = load_dataset('bigbio/pubmed_qa', 'pubmed_qa_labeled_fold0_source')
        self.formatted_data = {}

        for split in ['train', 'validation', 'test']:
            formatted_split_data = []

            for data in self.dataset[split]:
                question = data['QUESTION']
                contexts = ' '.join(data['CONTEXTS'])
                reasoning_required_pred = data['reasoning_required_pred']
                reasoning_free_pred = data['reasoning_free_pred']
                final_decision = data['final_decision']
                long_answer = data['LONG_ANSWER']
                formatted_split_data.append([question, contexts, final_decision, long_answer, reasoning_required_pred, reasoning_free_pred])

            self.formatted_data[split] = formatted_split_data

    def preprocess_data(self):

        print("Preprocessing Dataset...")

        self.tokenizer.pad_token = 0
        self.tokenizer.padding_side = 'left'
        self.inputs = {}

        self.max_length = 1300
        for split in ['train', 'validation', 'test']:
            self.inputs[split] = []
            labels = []
            
            for item in self.formatted_data[split]:
                question, contexts, final_decision, long_answer, reasoning_required_pred, reasoning_free_pred = item

                # Here we add few shot examples to each of the items in the test split, and base this off of the reasoning required and reasoning free flags 
                if split == 'test':

                    # FEW SHOT EXAMPLES
                    qa_examples = "Question: Can communication with terminally ill patients be taught? Answer: yes. Question: Does the sequence of clamp application during open abdominal aortic aneurysm surgery influence distal embolisation? Answer: no. Question: Do parents recall and understand children's weight status information after BMI screening? Answer: maybe."
                    qca_examples = "Question: Is the zeolite hemostatic agent beneficial in reducing blood loss during arterial injury? Context: Uncontrolled hemorrhage is the leading cause of fatality. The aim of this study was to evaluate the effect of zeolite mineral (QuikClot - Advanced Clotting Sponge [QC-ACS]) on blood loss and physiological variables in a swine extremity arterial injury model.", "Sixteen swine were used. Oblique groin incision was created and a 5 mm incision was made. The animals were allocated to: control group (n: 6): Pressure dressing was applied with manual pressure over gauze sponge; or QC group (n: 10): QC was directly applied over lacerated femoral artery. Mean arterial pressure, blood loss and physiological parameters were measured during the study period.", "Application of QC led to a slower drop in blood pressure. The control group had a significantly higher increase in lactate within 60 minutes. The mean prothrombin time in the control group was significantly increased at 60 minutes. The application of QC led to decreased total blood loss. The QC group had significantly higher hematocrit levels. QC application generated a significant heat production. There were mild edematous and vacuolar changes in nerve samples. Answer: yes. Question: Can nonproliferative breast disease and proliferative breast disease without atypia be distinguished by fine-needle aspiration cytology? Context: Cytologic criteria reported to be helpful in the distinction of proliferative breast disease without atypia (PBD) from nonproliferative breast disease (NPBD) have not been rigorously tested.", "Fifty-one air-dried, Diff-Quik-stained fine-needle aspirates (FNA) of palpable breast lesions with biopsy-proven diagnoses of NPBD (34 cases) or PBD (17 cases) were reviewed. The smears were evaluated for the cellularity, size, and architectural arrangement of the epithelial groups; the presence of single epithelial cells and myoepithelial cells; and nuclear characteristics.", "The only cytologic feature found to be significantly different between PBD and NPBD was a swirling pattern of epithelial cells. A swirling pattern was noted in 13 of 17 PBD cases (76%) and 12 of 34 NPBD cases (35%) (P = 0.008). Answer: no. Question: Does shaving the incision site increase the infection rate after spinal surgery? Context: A prospective randomized clinical study.", "To determine whether shaving the incision site before spinal surgery causes postsurgical infection.", "Spine surgeons usually shave the skin of the incision site immediately before surgery is performed. However, evidence from some surgical series suggests that presurgical shaving may increase the postsurgical infection rate. To our knowledge, no previously published studies have addressed this issue.", "A total of 789 patients scheduled to undergo spinal surgery were randomly allocated into 2 groups: those in whom the site of operation was shaved immediately before surgery (shaved group; 371 patients) and the patients in whom presurgical shaving was not performed (unshaved group; 418 patients). The mean duration of anesthesia and the infection rates in both groups were recorded and compared.", "The duration of anesthesia did not differ in the 2 groups (P>0.05). A postoperative infection developed in 4 patients in the shaved group and in 1 patient in the nonshaved group (P<0.01). Answer: maybe."
                    qlaa_examples = "Question: Is the affinity column-mediated immunoassay method suitable as an alternative to the microparticle enzyme immunoassay method as a blood tacrolimus assay? Context: The ACMIA method used for a tacrolimus assay is precise and has advantages, including the lack of a required pretreatment procedure. Furthermore, it is only slightly influenced by the hematologic or biochemical status of the samples. Answer: yes. Question: Does strategy training reduce age-related deficits in working memory? Context: Strategy training can boost WM performance, and its benefits appear to arise from strategy-specific effects and not from domain-general gains in cognitive ability. Answer: no. Question: Can calprotectin predict relapse risk in inflammatory bowel disease? Context: Measuring calprotectin may help to identify UC and colonic CD patients at higher risk of clinical relapse. Answer: maybe."
                    qclaa_examples = "Question: Does a physician's specialty influence the recording of medication history in patients' case notes? Context: To determine the impact of a physician's specialty on the frequency and depth of medication history documented in patient medical records.", "A cross-sectional assessment of the frequency and depth of medication history information documented by 123 physicians for 900 randomly selected patients stratified across Cardiology, Chest, Dermatology, Endocrine, Gastroenterology, Haematology, Neurology, Psychiatry and Renal specialties was carried out at a 900-bed teaching hospital located in Ibadan, Nigeria.", "Four hundred and forty-three (49.2%) of the cohort were males and 457 (50.8%) were females; with mean ages 43.2 +/- 18.6 and 43.1 +/- 17.9 years respectively. Physicians' specialties significantly influenced the depth of documentation of the medication history information across the nine specialties (P<0.0001). Post hoc pair-wise comparisons with Tukey's HSD test showed that the mean scores for adverse drug reactions and adherence to medicines was highest in the Cardiology specialty; while the Chest specialty had the highest mean scores for allergy to drugs, food, chemicals and cigarette smoking. Mean scores for the use of alcohol; illicit drugs; dietary restrictions was highest for Gastroenterology, Psychiatry and Endocrine specialties respectively. Physicians' specialties also significantly influenced the frequency of documentation of the medication history across the nine specialties (P<0.0001). Physicians appear to document more frequently and in greater depth medication history information that may aid the diagnostic tasks in their specific specialty. Researchers and other users of medication history data documented in patients' medical records by physicians may want to take special cognizance of this phenomenon. Answer: yes. Question: Are patients with diabetes receiving the same message from dietitians and nurses? Context: The purpose of this study was to determine if registered dietitian (RD) and registered nurse (RN) certified diabetes educators (CDEs) provide similar recommendations regarding carbohydrates and dietary supplements to individuals with diabetes.", "A survey was mailed to CDEs in the southern United States. Participants were asked to indicate their recommendations for use of carbohydrates, fiber, artificial sweeteners, and 12 selected dietary and herbal supplements when counseling individuals with diabetes.", "The survey sample consisted of 366 CDEs: 207 were RNs and 159 were RDs. No statistically significant differences were found between RNs and RDs in typical carbohydrate recommendations for treatment of diabetes. However, RDs were more likely than RNs to make recommendations for fiber intake or use of the glycemic index. A significant difference also was found in the treatment of hypoglycemia: RNs were more likely than RDs to recommend consuming a carbohydrate source with protein to treat hypoglycemia. Although some differences existed, RD and RN CDEs are making similar overall recommendations in the treatment of individuals with diabetes. Answer: no. Question: Does the severity of obstructive sleep apnea predict patients requiring high continuous positive airway pressure? Context: To investigate polysomnographic and anthropomorphic factors predicting need of high optimal continuous positive airway pressure (CPAP).", "Retrospective data analysis.", "Three hundred fifty-three consecutive obstructive sleep apnea (OSA) patients who had a successful manual CPAP titration in our sleep disorders unit.", "The mean optimal CPAP was 9.5 +/- 2.4 cm H2O. The optimal CPAP pressure increases with an increase in OSA severity from 7.79 +/- 2.2 in the mild, to 8.7 +/- 1.8 in the moderate, and to 10.1 +/- 2.3 cm H2O in the severe OSA group. A high CPAP was defined as the mean + 1 standard deviation (SD;>or =12 cm H2O). The predictor variables included apnea-hypopnea index (AHI), age, sex, body mass index (BMI), Epworth Sleepiness Scale (ESS), and the Multiple Sleep Latency Test (MSLT). High CPAP was required in 2 (6.9%), 6 (5.8%), and 63 (28.6%) patients with mild, moderate, and severe OSA, respectively. On univariate analysis, AHI, BMI, ESS score, and the proportion of males were significantly higher in those needing high CPAP. They also have a lower MSLT mean. On logistic regression, the use of high CPAP was 5.90 times more frequent (95% confidence interval 2.67-13.1) in severe OSA patients after adjustment for the other variables. The area under the receiver operator curve was 72.4%, showing that the model was adequate. Severe OSA patients are much more likely to need high CPAP levels. However, because of the low positive predictive value (only 28.6%), the clinical value of such information is limited. ESS and MSLT did not increase the predictive value for the need for high CPAP. Answer: maybe."

                    input_sequence = " "
                    if (reasoning_required_pred == "yes") and (reasoning_free_pred == "yes"):
                        input_sequence += ''.join(qclaa_examples)
                        input_sequence += " Question: " + question
                        input_sequence += " Context: " + contexts
                        input_sequence += " " + long_answer
                        input_sequence += " Answer: "

                    elif reasoning_required_pred == "yes":
                        input_sequence += ''.join(qca_examples)
                        input_sequence += " Question: " + question
                        input_sequence += " Context: " + contexts
                        input_sequence += " Answer: "

                    elif reasoning_free_pred == "yes":
                        input_sequence += ''.join(qlaa_examples)
                        input_sequence += " Question: " + question
                        input_sequence += " Context: " + long_answer
                        input_sequence += " Answer: "

                    else:
                        input_sequence += ''.join(qa_examples)
                        input_sequence += " Question: " + question
                        input_sequence += " Answer: "
                    
                    '''
                    ZERO SHOT
                    input_sequence = question
                    if reasoning_required_pred == "yes":
                        input_sequence += " " + contexts
                    if reasoning_free_pred == "yes":
                        input_sequence += " " + long_answer
                    '''
                else:
                    # For train and validation data
                    input_sequence = question
                    if reasoning_required_pred == "yes":
                        input_sequence += " " + contexts
                    if reasoning_free_pred == "yes":
                        input_sequence += " " + long_answer
                    input_sequence += " Answer: "

                
                label_sequence = final_decision

                encoded_inputs = self.tokenizer.encode_plus(
                    input_sequence,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    pad_to_max_length=True
                )

                encoded_labels = self.tokenizer.encode_plus(
                    label_sequence,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    pad_to_max_length=True
                )

                self.inputs[split].append(encoded_inputs['input_ids'])
                labels.append(encoded_labels['input_ids'])

            self.inputs[split] = torch.cat(self.inputs[split], dim=0)
            self.inputs[split + 'labels'] = torch.cat(labels, dim=0)


    def data_loader(self):

        print("Loading Preprocessed Data...")

        for split in ['train', 'validation', 'test']:
            dataset = torch.utils.data.TensorDataset(self.inputs[split], self.inputs[split + 'labels'])
            
            if split == 'train':
                self.train_dataloader = DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=self.batch_size,
                    pin_memory=True
                )
            elif split == 'validation':
                self.eval_dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    pin_memory=True
                )
            elif split == 'test':
                self.test_dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    pin_memory=True
                )


    def train_model(self):

        print("Training...")

        self.train_loss_values = []
        self.val_loss_values = []
        self.test_loss_values = []
        self.f1_scores = []
        self.accuracy_scores = []

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=(len(self.train_dataloader) * self.num_epochs))

        self.model = self.model.to(self.device)

        # Training Loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            # Training
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs, labels=labels)

                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            self.train_loss_values.append(total_loss / len(self.train_dataloader))

            # Validation
            self.model.eval()
            eval_loss = 0
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                eval_loss += loss.detach().float()
            self.val_loss_values.append(eval_loss / len(self.eval_dataloader))

            '''
            # Test 
            # Note that I was using this code to plot the test metrics, but this leaked test data into training and effected the results
            test_loss = 0
            for step, batch in enumerate(tqdm(self.test_dataloader)):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                test_loss += loss.detach().float()
            self.test_loss_values.append(test_loss / len(self.test_dataloader)) 
            '''

            eval_epoch_loss = eval_loss / len(self.eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(self.train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)

            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
            self.log(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        

    def classify_text(self, text):
    
        # Create analyzer
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)

        # Label the model's output as either yes, no, or maybe based on sentiment analysis 
        if sentiment['compound'] >= 0.45:
            return 'yes'
        elif sentiment['compound'] <= -0.45:
            return 'no'
        else:
            return 'maybe'
        
       
    
    def infer(self):

        '''
        # This code pushes the model to hugging face library

        write_token = "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"
        login(token=write_token)
        #print("Successfully logged in...")

        peft_model_id = "test-model-peft"
        api = HfApi()
        api.delete_repo(repo_id=peft_model_id)
        #print("Successfully deleted the previous repo...")

        api.create_repo(repo_id=peft_model_id, private=True)
        #print("Successfully created the repo...")

        self.model.push_to_hub(peft_model_id, use_auth_token=write_token)
        #print("Successfully pushed to the hub...")

        peft_model_id = "Ryan-Engel/test-model-peft"

        config = PeftConfig.from_pretrained(peft_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        '''

        # Download the vader lexicon
        nltk.download('vader_lexicon')

        self.model.to(self.device)
        self.model.eval()

        model_preds = []
        true_labels = []

        for step, batch in enumerate(self.test_dataloader):
            
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():

                # Get model output
                outputs = self.model(inputs, labels=labels)
                
                # Compute softmax over the logits to get probabilities./run.
                probs = F.softmax(outputs.logits, dim=-1)

                # We get the top k possible outputs for more diversity (Changed this value for testing and optimizing hyper-parameters)
                k = 1
                top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)

                # Choose from top k
                chosen_indices = torch.multinomial(top_k_probs.reshape(-1, k), num_samples=1)
                chosen_indices = chosen_indices.reshape(*top_k_probs.shape[:-1])

                # Use the chosen indices to gather the tokens
                preds = top_k_indices.gather(-1, chosen_indices.unsqueeze(-1)).squeeze(-1)
                pred_texts = self.tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)

                # Extract the true labels
                true_texts = self.tokenizer.batch_decode(labels.detach().cpu().numpy(), skip_special_tokens=True)

                for pred_text, true_text, input in zip(pred_texts, true_texts, inputs):

                    # Replace series of zeros with a space, eliminates padding from text
                    pred_text = re.sub('0+', ' ', pred_text)
                    
                    # Strip leading and trailing spaces and return
                    pred_text = pred_text.strip()

                    # Remove newline characters 
                    pred_text = pred_text.replace('\n', '')

                    # Classify text using sentiment analysis function
                    pred_classification =  self.classify_text(pred_text)
                    model_preds.append(pred_classification)

                    # Debug
                    #print("TRUE TEXT: ", true_text)

                    # Get result from true label
                    if 'yes' in true_text.lower():
                        true_classification = 'yes'
                    elif 'no' in true_text.lower():
                        true_classification = 'no'
                    else:
                        true_classification = 'maybe'
                    true_labels.append(true_classification)

                    # Debug
                    #self.log(f"INPUT: {''.join(self.tokenizer.batch_decode(input.detach().cpu().numpy(), skip_special_tokens=True))}\n")
                    #self.log(f"PREDICTED TEXT: {pred_text}\n")
                    #self.log(f"PREDICTED CLASSIFICATION: {pred_classification}\n")
                    #self.log(f"TRUE CLASSIFICATION: {true_classification}\n")

        # Calculate frequencies for model_preds
        pred_counter = Counter(model_preds)
        pred_total = len(model_preds)
        pred_percentages = {label: (count / pred_total) * 100 for label, count in pred_counter.items()}

        # Calculate frequencies for true_labels
        true_counter = Counter(true_labels)
        true_total = len(true_labels)
        true_percentages = {label: (count / true_total) * 100 for label, count in true_counter.items()}

        # Display frequencies
        print("Model Predictions Percentages: ", pred_percentages)
        print("True Labels Percentages: ", true_percentages)
        self.log(f"Model Predictions Percentages: {pred_percentages}")
        self.log(f"True Labels Percentages: {true_percentages}")


        # Compute F1 score and accuracy
        f1 = f1_score(true_labels, model_preds, average='weighted')
        accuracy = accuracy_score(true_labels, model_preds)

        # Display metrics
        self.log(f"F1 Score: {f1}")
        self.log(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"Accuracy: {accuracy}")


    def plot_metrics(self):

        train_loss_values_np = [t.cpu().numpy() for t in self.train_loss_values]
        val_loss_values_np = [t.cpu().numpy() for t in self.val_loss_values]
        #test_loss_values_np = [t.cpu().numpy() for t in self.test_loss_values]

        plt.figure(figsize=(10, 5))

        epochs = range(1, len(train_loss_values_np) + 1)

        # Plot the training, validation, and test losses
        plt.plot(epochs, train_loss_values_np, 'b', label='Training loss')
        plt.plot(epochs, val_loss_values_np, 'b', label='Validation loss')
        #plt.plot(epochs, test_loss_values_np, 'b', label='Test loss') 

        plt.title('Training, Validation and Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('metrics.png')







        
                




