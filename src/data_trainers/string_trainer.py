#Necessary Imports
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig, LoraConfig
from huggingface_hub import login, HfApi
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
import random
from sklearn.model_selection import train_test_split


class string_trainer():

    def init(self, model, tokenizer, log_file, data_processor):

        print("Initializing String Peft Trainer...")

        self.log_file = log_file
        open(self.log_file, 'w').close()

        self.device = "cuda"
        self.model_name_or_path = model
        self.tokenizer = tokenizer
        self.data_processor = data_processor
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        
        # PROMPT TUNING CONFIG
        # Optimal results with 10 epochs, lr = .001 and batch size 4, I used random initialization because for this task because using random yielded higher results than any text prompt
        #self.peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, prompt_tuning_init=PromptTuningInit.RANDOM, num_virtual_tokens=8, tokenizer_name_or_path=self.model_name_or_path)
        #self.model = get_peft_model(self.model, self.peft_config)
        
        # LORA CONFIG
        # Optimal results with 10 epochs, lr = .001 and batch size = 4
        self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=8, lora_dropout=.01, bias="lora_only")
        self.model = get_peft_model(self.model, self.peft_config)

        # For combined strategy, optimal results with 3 epochs, lr = .001 and batch size = 4

        self.lr = .001
        self.num_epochs = 10
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

        #print("Loading Dataset...")

        data = self.data_processor.data_reader.train_data
        model_prompt = self.data_processor.get_model_prompt()
        
        n = 1000

        # List to hold questions/prompts and answers
        self.questions = []
        self.answers = []

        keys = list(data.keys()) #19,382

        # Shuffle data to get random samples
        random.shuffle(keys)

        count = 0 # Counter to keep track of number of samples generated
        for key in keys:

            # Construct and append the prompt/question and corresponding answer
            self.questions.append(model_prompt['entity_q'](key))
            self.answers.append(model_prompt['entity_a'](", ".join(sorted(list(set(data[key]))))))

            count += 1
            # Stop when we have enough samples
            if count >= n:
                break


    def preprocess_data(self):

        print("Preprocessing Dataset...")

        self.tokenizer.pad_token = 0
        self.tokenizer.padding_side = 'left'

        # Split the data into train (70%), validation (15%), and test data (15%)
        train_q, temp_q, train_a, temp_a = train_test_split(self.questions, self.answers, test_size=0.3, random_state=42)
        valid_q, test_q, valid_a, test_a = train_test_split(temp_q, temp_a, test_size=0.5, random_state=42)

        self.inputs = {'train': [], 'valid': [], 'test': []}
        self.labels = {'train': [], 'valid': [], 'test': []}

        self.max_length = 1700

        for split, (q_split, a_split) in zip(['train', 'valid', 'test'], [(train_q, train_a), (valid_q, valid_a), (test_q, test_a)]):
            input_ids = []
            label_ids = []
            for question, answer in zip(q_split, a_split):

                if split == 'test':

                    #FEW SHOT EXAMPLE
                    question = "Question: Which proteins are related to KRT81?\n\nAnswer:  AATF, ACSL6, ACTL6B, ADRA1D, AGL, AMER2, AMER3, AVPR1B, B3GNT8, BCS1L, BICD1, BICD2, C16orf45, C8orf46, CA1, CABP1, CACNA1A, CACNA1B, CACNA1I, CEACAM21, CEACAM3, CEACAM4, CEACAM5, CEACAM6, CKB, CKM, CMTM5, CNFN, CNIH3, CNPPD1, DYDC1, DYDC2, DZANK1, ELAVL3, ELMOD1, ENSG267881, ENSG268361, ENSG2685, ENTPD1, ENTPD2, ENTPD3, ENTPD8, ETHE1, FAM131B, FAM155A, FAM163B, FXYD2, GABRA1, GABRA6, GABRB1, GABRD, GRM1, GRM4, GSK3A, HAGH, HAGHL, HAPLN2, HHEX, HIC2, HSPB9, HTR5A, IGF2, IGLL5, IGSF11, KBTBD6, KBTBD7, KCNC1, KCNC2, KCNC4, KCNH4, KCNJ9, KIFC1, KIFC2, KIFC3, LACTB2, LIPE, LIPJ, LMO3, LONRF2, LRRC24, LRRC4, MAG, MAP7D2, MAPK8IP2, MAPT, NUDC, NUDCD2, NUDCD3, NWD2, NXPH1, OBBP2, OPALIN, OR8K3, PAFAH1B2, PAFAH1B3, PARVB, PCDH17, PCDH9, PCDHGA8, PCSK2, PGBD5, PI4KB, PNKD, PNMAL1, PRKCE, PRKD3, PRR18, PRR19, PSG1, PSG11, PSG2, PSG3, PSG4, PSG6, PSG7, PSG8, PSG9, PSMC3IP, PTCRA, PTER, RAB9B, RASA3, RBFOX3, REEP2, RGS7BP, RNF112, RNF157, RNFT2, RPH3A, RUNDC3A, SAMD5, SCG3, SCN1A, SCN2B, SCRT1, SEZ6L, SGK2, SGTB, SH2B3, SH2D4B, SHISA7, SIGLEC1, SIGLEC11, SIGLEC8, SIGLEC9, SLC12A5, SLC13A4, SLC13A5, SLC8A2, SLITRK4, SLX4IP, SNAP25, SNAP91, SNCB, SORCS3, SPON1, SPON2, ST8SIA3, STMN4, STX1B, STXBP5L, SUGP1, SULT4A1, SVOP, SYP, SYT1, SYT16, SYT4, TBC1D14, TBC1D16, TBC1D5, TBCEL, TCEAL6, TGDS, TGFB1, TMEM13, TMEM151B, TMEM178A, TMEM179, TMEM231, TMEM26, TMEM59L, TMEM88B, TMEM91, TNNI3, TNR, TOMM4, TOMM4L, TRNP1, TTC9B, TTYH1, TTYH3, TUBB4A, UNCX, USP17L11, USP9X, VSIG2, VSTM2A, VSTM2B, XRCC1, YME1L1, ZFP64, ZMAT4, ZNF267, ZNF296, ZNF526, ZNF574, ZNF75D</s>\n\n" + question

                encoded_question = self.tokenizer.encode_plus(
                    question,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                encoded_answer = self.tokenizer.encode_plus(
                    answer,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                input_ids.append(encoded_question['input_ids'])
                label_ids.append(encoded_answer['input_ids'])

            self.inputs[split] = torch.cat(input_ids, dim=0)
            self.labels[split] = torch.cat(label_ids, dim=0)


    def data_loader(self):

        print("Loading Preprocessed Data...")

        for split in ['train', 'valid', 'test']:

            # Ensure that the inputs and labels are in Tensor format
            input_ids = self.inputs[split] if isinstance(self.inputs[split], torch.Tensor) else torch.stack(self.inputs[split])
            label_ids = self.labels[split] if isinstance(self.labels[split], torch.Tensor) else torch.stack(self.labels[split])
            
            dataset = torch.utils.data.TensorDataset(input_ids, label_ids)
            
            if split == 'train':
                self.train_dataloader = DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=self.batch_size,
                    pin_memory=True
                )
            elif split == 'valid':
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
                labels = torch.cat([labels[:, 1:], labels.new_full((labels.size(0), 1), self.tokenizer.pad_token_id)], dim=1)

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
            eval_preds = []
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = torch.cat([labels[:, 1:], labels.new_full((labels.size(0), 1), self.tokenizer.pad_token_id)], dim=1)

                with torch.no_grad():
                    outputs = self.model(inputs, labels=labels)

                loss = outputs.loss
                eval_loss += loss.detach().float()

                eval_preds.extend(self.tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

            self.val_loss_values.append(eval_loss / len(self.eval_dataloader))
            eval_epoch_loss = eval_loss / len(self.eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(self.train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)

            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
            self.log(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


    def infer(self):

        '''
        # This code pushes the model to hugging face library

        write_token = "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"
        login(token=write_token)

        peft_model_id = "string-prompt-tuning-model"
        api = HfApi()
        api.delete_repo(repo_id=peft_model_id)

        api.create_repo(repo_id=peft_model_id, private=True)

        self.model.push_to_hub(peft_model_id, use_auth_token=write_token)

        peft_model_id = "Ryan-Engel/string-prompt-tuning-model"

        config = PeftConfig.from_pretrained(peft_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        '''

        self.model.to(self.device)
        self.model.eval()

        test_loss = 0
        test_true_tokens_batches = []
        test_preds_tokens_batches = []

        for step, batch in enumerate(tqdm(self.test_dataloader)):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Identify the padding tokens
            active_loss = labels != self.tokenizer.pad_token_id

            with torch.no_grad():
                outputs = self.model(inputs, labels=labels)

            loss = outputs.loss
            test_loss += loss.detach().float()

            k = 3  # Change value when testing
            probs = F.softmax(outputs.logits, dim=-1)

            # Get batch size and sequence length
            batch_size, seq_length = probs.shape[:2]

            # Flatten the probabilities to 2D (batch * sequence, vocab) before applying topk
            top_k_probs, top_k_indices = torch.topk(probs.view(-1, probs.shape[-1]), k, dim=-1)

            # Normalize top k probabilities
            top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)

            # Sample from the multinomial distribution
            top_k_sampled_indices = torch.multinomial(top_k_probs, 1)

            # Get the corresponding top k token indices
            predicted_tokens = top_k_indices.gather(-1, top_k_sampled_indices)

            # Reshape to the original size
            predicted_tokens = predicted_tokens.view(batch_size, seq_length)

            # Transfer to CPU
            predicted_tokens = predicted_tokens.detach().cpu()

            # Iterate over each sample in the batch
            for sample_idx in range(inputs.shape[0]):

                # Get active loss to calculate token level precision
                sample_active_loss = active_loss[sample_idx].cpu()
                sample_predicted_tokens = predicted_tokens[sample_idx].cpu()
                sample_labels = labels[sample_idx].cpu()
                sample_inputs = inputs[sample_idx].cpu()

                # Determine the maximum length between sample_active_loss and sample_predicted_tokens
                max_len = max(sample_predicted_tokens.shape[0], sample_active_loss.shape[0])

                # If sample_active_loss is smaller, pad it
                if sample_active_loss.shape[0] < max_len:
                    padding_size = max_len - sample_active_loss.shape[0]
                    sample_active_loss = F.pad(sample_active_loss.float(), (0, padding_size)).bool()

                # If other tensors are smaller, pad them as well
                if sample_predicted_tokens.shape[0] < max_len:
                    padding_size = max_len - sample_predicted_tokens.shape[0]
                    sample_predicted_tokens = F.pad(sample_predicted_tokens, (0, padding_size))

                if sample_labels.shape[0] < max_len:
                    padding_size = max_len - sample_labels.shape[0]
                    sample_labels = F.pad(sample_labels, (0, padding_size))

                if sample_inputs.shape[0] < max_len:
                    padding_size = max_len - sample_inputs.shape[0]
                    sample_inputs = F.pad(sample_inputs, (0, padding_size))

                # Now apply the mask
                sample_predicted_tokens = sample_predicted_tokens[sample_active_loss]
                sample_labels = sample_labels[sample_active_loss]

                # Append to the batch arrays
                test_preds_tokens_batches.append(sample_predicted_tokens)
                test_true_tokens_batches.append(sample_labels)

                # Decode and print the input, output, and label
                decoded_input =  re.sub(r'0+', '', (self.tokenizer.decode(sample_inputs, skip_special_tokens=True).strip()))
                decoded_output = re.sub(r'0+', '', (self.tokenizer.decode(sample_predicted_tokens, skip_special_tokens=True).strip()))
                decoded_label = re.sub(r'0+', '', (self.tokenizer.decode(sample_labels, skip_special_tokens=True).strip()))


                # The following code is used to filter nonsensical output, this strategy may not be the most effective
                # The reasoning for doing this is because the model generalizes to outputs of long sequences of "1" or "," or other combinations
                # After applying these filters, the model then begins to generalize to other sequences, indicating that a better approach is needed

                # Find the index of the first newline character
                end_index = decoded_input.find('\n')

                # Extract the string until the first newline character
                prompt = decoded_input[:end_index]

                # Replace sequences of comma-space-comma with a single comma
                decoded_output = re.sub(', ,', ',', decoded_output)

                # Replace sequences of comma-space-comma with a single comma
                decoded_output = re.sub(',1', '', decoded_output)

                # Remove sequences of "1," or "11," etc.
                decoded_output = re.sub('1+,', '', decoded_output)

                # Remove sequences of commas ",,"
                decoded_output = re.sub(',+', ',', decoded_output)

                # Clean Response
                text_list = self.data_processor.clean_response(decoded_output, prompt)

                # Iterate through each item in the list
                for text in text_list:
                    # Re-encode the cleaned output
                    re_encoded_output = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids[0]

                    # Ensure the re-encoded output is in the same device as the model
                    re_encoded_output = re_encoded_output.to(self.device)
                
                    # Append re_encoded_output to test_preds_tokens_batches
                    test_preds_tokens_batches.append(re_encoded_output)

                test_true_tokens_batches.append(sample_labels)

                # Debug
                #self.log(f"INPUT: {decoded_input}\n")
                #self.log(f"OUTPUT: {decoded_output}\n")
                #self.log(f"TRUE LABEL: {decoded_label}\n")

                #print(f"INPUT: {decoded_input}\n")
                #print(f"OUTPUT: {decoded_output}\n")
                #print(f"TRUE LABEL: {decoded_label}\n")

        test_preds_tokens = [token.item() for batch in test_preds_tokens_batches for token in batch]
        test_true_tokens = [token.item() for batch in test_true_tokens_batches for token in batch]

        correct_tokens = 0
        total_predicted_tokens = len(test_preds_tokens)
        for pred_token, true_token in zip(test_preds_tokens, test_true_tokens):
            if pred_token == true_token:
                correct_tokens += 1

        if total_predicted_tokens > 0:  # Avoid division by zero
            precision = correct_tokens / total_predicted_tokens
        else:
            precision = 0.0

        print("Total Tokens: ", total_predicted_tokens)
        print(f"Precision at token-level: {precision}")
        self.log(f"Precision at token-level: {precision}")







