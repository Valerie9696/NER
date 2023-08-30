import os

import numpy as np
from sklearn.metrics import accuracy_score
import torch
import preproccessing as prep
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch import cuda
from seqeval.metrics import classification_report
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }


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
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len({**self.train_dataset.id2label, **self.test_dataset.id2label}), id2label={**self.train_dataset.id2label,**self.test_dataset.id2label}, label2id={**self.train_dataset.label2id,**self.test_dataset.label2id})
        self.model.to(device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            print(f"Training epoch: {epoch + 1}")
            self.train(epoch)
        self.validate()
        self.model.save_pretrained(os.path.join('models', 'bert_pretrained.mdl'))

    def train(self, epoch):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        self.model.train()

        for idx, batch in enumerate(self.train_loaded):

            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            outputs = self.model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

            # compute training accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(input=active_logits, dim=1)  # prev axis = 1 shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_preds.extend(predictions)
            tr_labels.extend(targets)

            tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=MAX_GRAD_NORM
            )

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    def validate(self):
        # put model in evaluation mode
        self.model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
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
                nb_eval_examples += targets.size(0)

                if idx % 100 == 0:
                    loss_step = eval_loss / nb_eval_steps
                    print(f"Validation loss per 100 evaluation steps: {loss_step}")

                # compute evaluation accuracy
                flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
                active_logits = eval_logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(input=active_logits, dim=1)  # shape (batch_size * seq_len,)
                # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
                active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
                targets = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)

                eval_labels.extend(targets)
                eval_preds.extend(predictions)

                tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
                eval_accuracy += tmp_eval_accuracy

        # print(eval_labels)
        # print(eval_preds)
        fullid2label = {self.train_dataset.id2label,self.test_dataset.id2label}
        tags = [fullid2label[id.item()] for id in eval_labels]
        predictions = [fullid2label[id.item()] for id in eval_preds]

        # print(labels)
        # print(predictions)

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}")

        print(classification_report([tags], [predictions]))

        return tags, predictions