import os
import gc
import numpy as np
from sklearn.metrics import f1_score
import torch
import preproccessing as prep
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch import cuda
from seqeval.metrics import classification_report
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb

TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 0.001#1e-05
MAX_NORM = 10
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 3}
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 3}

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None

        if patience == 0:
            self.is_better = True
            self.step = lambda a: False

    def step(self, loss):
        if self.best is None:
            self.best = loss
            return False
        if np.isnan(loss):
            return True
        #print(loss, self.best, self.num_bad_epochs)
        #print('is_better', self.is_better(metrics, self.best))
        change = self.check(loss, min_delta=10)
        if change:
            self.num_bad_epochs = 0
            self.best = loss
        else:
            self.num_bad_epochs += 1
        print('count of bad epochs', self.num_bad_epochs)
        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True
        return False

    def check(self, loss, min_delta):
        print(self.best - loss, min_delta)
        change = False
        if (self.best - loss) > min_delta:
            self.is_better = True
            change = True
        else:
            self.is_better = False
            change = False
        return change


def tokenize_and_preserve_labels(sentence=None, tags=None, tokenizer=tokenizer):
    tokenized_sentence = []
    tags = []
    #sentence = sentence.strip()
    for word, tag in zip(sentence, tags):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized= tokenizer.tokenize(word)
        sub_words = len(tokenized)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized)

        # Add the same label to the new list of labels `n_subwords` times
        tags.extend([tag] * sub_words)

    return tokenized_sentence, tags


class BertPrepper(Dataset):
    def __init__(self, sentences, tags, unique_tags, tokenizer, max_len):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = len(sentences)
        self.label2id = {k: v for v, k in enumerate(unique_tags)}
        self.id2label = {v: k for v, k in enumerate(unique_tags)}

    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.sentences[index]
        word_labels = self.tags[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)

        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]  # add special tokens
        labels.insert(0, "O")  # add outside label for [CLS] token
        labels.insert(-1, "O")  # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
            # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            labels = labels[:maxlen]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(maxlen - len(tokenized_sentence))]
            labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [self.label2id[label] for label in labels]
        # the following line is deprecated
        # label_ids = [label if label != 0 else -100 for label in label_ids]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len


class Model:
    def __init__(self):
        self.data = prep.Dataloader(train_path='train.json', test_path='test.json')
        self.train_dataset = BertPrepper(sentences=self.data.train_sentences, tags=self.data.train_tags, unique_tags=self.data.unique_tags, tokenizer=tokenizer, max_len=128)
        self.test_dataset = BertPrepper(sentences=self.data.test_sentences, tags=self.data.test_tags, unique_tags=self.data.unique_tags, tokenizer=tokenizer, max_len=128)
        self.train_loaded = DataLoader(self.train_dataset, **train_params)
        self.test_loaded = DataLoader(self.test_dataset, **test_params)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # print('allocated memory: ', torch.cuda.memory_allocated(device=device))
            print('before grad_clipping: free and total memory: ',
                  torch.cuda.mem_get_info(device=torch.cuda.current_device()))
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len({**self.train_dataset.id2label, **self.test_dataset.id2label}), id2label={**self.train_dataset.id2label,**self.test_dataset.id2label}, label2id={**self.train_dataset.label2id,**self.test_dataset.label2id})
        self.model.to(device)
        self.epoch_stop = EarlyStopping(patience=5)
        self.early_stopping = EarlyStopping(patience=5)
        #self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            print(f"Training epoch: {epoch + 1}")
            epoch_loss = self.train(epoch)
            if self.epoch_stop.step(epoch_loss):
                print('tis enough')
                break
        self.validate()
        self.model.save_pretrained(os.path.join('models', 'bert_pretrained.mdl'))

    def get_f1_score(self, targets, logits, mask, training_predictions, training_labels, full_f1):
        flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(input=active_logits, dim=1)  # prev axis = 1 shape (batch_size * seq_len,)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        training_predictions.extend(predictions)
        training_labels.extend(targets)
        cur_f1 = f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average='micro')
        full_f1 += cur_f1
        return full_f1

    def train(self, epoch):
        # init variables
        train_loss = 0
        f1_train = 0
        train_step_count = 0
        training_predictions = []
        training_labels = []
        # start training
        self.model.train()
        for i, batch in enumerate(self.train_loaded):
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)
            result = self.model(input_ids=ids, attention_mask=mask, labels=targets)
            loss = result.loss
            logits = result.logits
            train_loss += loss.item()
            train_step_count += 1
            if i % 100 == 0:
                loss_step = train_loss/train_step_count
                print('Training: Loss per 100 steps: ', loss_step)
                if self.early_stopping.step(loss_step):
                    print('stop mid epoch')
                    break
            # get the f1-score
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                #print('allocated memory: ', torch.cuda.memory_allocated(device=device))
                print('before_f1: free and total memory: ', torch.cuda.mem_get_info(device=torch.cuda.current_device()))
            f1_train = self.get_f1_score(targets=targets, logits=logits, mask=mask, training_predictions=training_predictions, training_labels=training_labels, full_f1=f1_train)
            # gradient clipping
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                #print('allocated memory: ', torch.cuda.memory_allocated(device=device))
                print('before grad_clipping: free and total memory: ', torch.cuda.mem_get_info(device=torch.cuda.current_device()))
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=MAX_NORM)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                #print('allocated memory: ', torch.cuda.memory_allocated(device=device))
                print('before zero grad: free and total memory: ', torch.cuda.mem_get_info(device=torch.cuda.current_device()))
            # backward pass
            self.optimizer.zero_grad()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                #print('allocated memory: ', torch.cuda.memory_allocated(device=device))
                print('free and total memory: ', torch.cuda.mem_get_info(device=torch.cuda.current_device()))
                print(torch.cuda.memory_summary(device=device))
            loss.backward()
            print('after loss')
            self.optimizer.step()
            print('after step')
        final_loss = train_loss/train_step_count
        print(f"loss of epoch: {final_loss}")
        f1_train = f1_train/train_step_count
        print(f"f1-score of epoch: {f1_train}")
        return final_loss

    def validate(self):
        # put model in evaluation mode
        self.model.eval()

        eval_loss = 0
        f1_val = 0
        nb_eval_steps = 0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(self.test_loaded):
                ids = batch['ids'].to(device, dtype=torch.long)
                mask = batch['mask'].to(device, dtype=torch.long)
                targets = batch['targets'].to(device, dtype=torch.long)

                outputs = self.model(input_ids=ids, attention_mask=mask, labels=targets)
                loss, eval_logits = outputs.loss, outputs.logits

                eval_loss += loss.item()

                nb_eval_steps += 1
                if idx % 100 == 0:
                    loss_step = eval_loss / nb_eval_steps
                    print('Validation: Loss per 100 steps: ', loss_step)

                f1_val = self.get_f1_score(targets=targets, logits=eval_logits, mask=mask, training_predictions=eval_preds, training_labels=eval_labels, full_f1=f1_val)


        # print(eval_labels)
        # print(eval_preds)
        id2label_combined = {**self.train_dataset.id2label,  **self.test_dataset.id2label}
        tags = [id2label_combined[id.item()] for id in eval_labels]
        predictions = [id2label_combined[id.item()] for id in eval_preds]

        # print(labels)
        # print(predictions)

        eval_loss = eval_loss / nb_eval_steps
        final_f1 = f1_val / nb_eval_steps
        print(f"Validation Loss: {eval_loss}")
        print(f"Validation Accuracy: {final_f1}")

        print(classification_report([tags], [predictions]))

        return tags, predictions
