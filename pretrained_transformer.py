import os
import gc
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import cuda
import preproccessing as prep
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from seqeval.metrics import classification_report
from sklearn.metrics import f1_score as f1_score, accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = 'cuda' if cuda.is_available() else 'cpu'
#https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 3e-05
MAX_NORM = 10
NUM_WORKERS = 8
TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': False, 'num_workers': NUM_WORKERS}
VALID_PARAMS = {'batch_size': VALID_BATCH_SIZE, 'shuffle': False, 'num_workers': NUM_WORKERS}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': NUM_WORKERS}


class BertBase:
    def __init__(self, filter_dataset=False):
        self.data = prep.Dataloader(train_path='train.json', test_path='test.json', filter_dataset=filter_dataset, bio=False)
        self.train_dataset = prep.BertPrepper(sentences=self.data.train_sentences, tags=self.data.train_tags,
                                              pos_tags=self.data.train_pos_tags,
                                              dependencies=self.data.train_dependencies,
                                              unique_tags=self.data.unique_tags, max_len=128, all_pos=self.data.all_pos,
                                              all_deps=self.data.all_deps)
        self.valid_dataset = prep.BertPrepper(sentences=self.data.valid_sentences, tags=self.data.valid_tags,
                                              pos_tags=self.data.valid_pos_tags,
                                              dependencies=self.data.valid_dependencies,
                                              unique_tags=self.data.unique_tags,max_len=128, all_pos=self.data.all_pos,
                                              all_deps=self.data.all_deps)
        self.test_dataset = prep.BertPrepper(sentences=self.data.test_sentences, tags=self.data.test_tags,
                                             pos_tags=self.data.test_pos_tags, dependencies=self.data.test_dependencies,
                                             unique_tags=self.data.unique_tags, max_len=128, all_pos=self.data.all_pos,
                                             all_deps=self.data.all_deps)
        self.train_loaded = DataLoader(self.train_dataset, **TRAIN_PARAMS)
        self.valid_loaded = DataLoader(self.valid_dataset, **VALID_PARAMS)
        self.test_loaded = DataLoader(self.test_dataset, **TEST_PARAMS)
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len({**self.train_dataset.id2label, **self.test_dataset.id2label}), id2label={**self.train_dataset.id2label,**self.test_dataset.id2label}, label2id={**self.train_dataset.label2id,**self.test_dataset.label2id})
        self.model.to(device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            print('Training of epoch: ', epoch + 1)
            epoch_loss = self.train(epoch)
        self.validate()
        self.final_f1 = self.test()
        if not os.path.exists('models'):
            os.mkdir('models')
        self.model.save_pretrained(os.path.join('models', 'bert_pretrained.h5'))

    def get_metric(self, targets, logits, mask, predictions, tags, full_val, metric='accuracy'):
        flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)        # shape (batch_size * seq_len, tag_count)
        flattened_predictions = torch.argmax(input=active_logits, dim=1)  # prev axis = 1 shape (batch_size * seq_len,)
        active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        preds = torch.masked_select(flattened_predictions, active_accuracy)
        predictions.extend(preds)
        tags.extend(targets)
        val = 0
        if metric == 'accuracy':
            val = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
        elif metric == 'micro_f1':
            val = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='micro')
        elif metric == 'macro_f1':
            val = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='macro')
        full_val += val
        return full_val, predictions, tags

    def train(self, epoch):
        # Initialize
        train_loss = 0
        acc_train = 0
        train_steps = 0
        predictions = []
        tags = []
        self.model.train()                                  # actual start of the training
        for i, batch in enumerate(self.train_loaded):
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)
            out = self.model(input_ids=ids, attention_mask=mask, labels=targets)
            loss = out.loss
            logits = out.logits
            train_loss += loss.item()
            train_steps += 1
            if i % 100 == 0:
                loss_step = train_loss/train_steps
                print('Training: Loss per 100 steps: ', loss_step)
            # get the f1-score
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            acc_train, predictions, tags = self.get_metric(targets=targets, logits=logits, mask=mask, predictions=predictions, tags=tags, full_val=acc_train, metric='accuracy')
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=MAX_NORM)
            # backward pass
            self.optimizer.zero_grad()
            gc.collect()
            loss.backward()
            self.optimizer.step()
        final_loss = train_loss/train_steps
        print('Loss of epoch: ', final_loss)
        full_acc = acc_train/train_steps
        print('Accuracy of epoch: ', full_acc)
        return final_loss

    def validate(self):
        self.model.eval()
        eval_loss = 0
        acc_val = 0
        val_steps = 0
        val_preds, val_tags = [], []
        with torch.no_grad():
            for idx, batch in enumerate(self.valid_loaded):
                ids = batch['ids'].to(device, dtype=torch.long)
                mask = batch['mask'].to(device, dtype=torch.long)
                targets = batch['targets'].to(device, dtype=torch.long)
                out = self.model(input_ids=ids, attention_mask=mask, labels=targets)
                loss = out.loss
                eval_logits = out.logits
                eval_loss += loss.item()
                val_steps += 1
                if idx % 100 == 0:
                    loss_step = eval_loss / val_steps
                    print('Validation: Loss per 100 steps: ', loss_step)

                acc_val, val_preds, val_tags = self.get_metric(targets=targets, logits=eval_logits, mask=mask, predictions=val_preds, tags=val_tags, full_val=acc_val, metric='accuracy')
        id2label_combined = self.train_dataset.id2label
        tags = [id2label_combined[id.item()] for id in val_tags]
        predictions = [id2label_combined[id.item()] for id in val_preds]
        eval_loss = eval_loss / val_steps
        final_acc = acc_val / val_steps
        print('Validation Loss: ', eval_loss)
        print('Validation Accuracy: ', final_acc)

        return final_acc

    def test(self):
        self.model.eval()
        f1_test = 0
        samples = 0
        test_preds = []
        test_tags = []
        # move to gpu
        for idx, batch in enumerate(self.test_loaded):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch['targets'].to(device, dtype=torch.long)
            out = self.model(input_ids=ids, attention_mask=mask)
            logits = out[0]
            active_logits = logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size*seq_len,) - predictions at the token level
            #tokens = self.test_dataset.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
            token_predictions = [self.test_dataset.id2label[i] for i in flattened_predictions.cpu().numpy()]
            #wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)
            f1_test, test_preds, test_tags = self.get_metric(targets=targets, logits=logits, mask=mask,
                                                            predictions=test_preds, tags=test_tags, full_val=f1_test, metric='micro_f1')
            word_level_predictions = []
            #for pair in wp_preds:
             #   if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                    # skip p
            # rediction
              #      continue
               # else:
                #    word_level_predictions.append(pair[1])
            samples += 1

        # we join tokens, if they are not special ones
            #str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
            #print(str_rep)
            #print(word_level_predictions)
        id2label_combined = self.train_dataset.id2label  # {**self.train_dataset.id2label, **self.valid_dataset.id2label, **self.test_dataset.id2label}
        tags = [id2label_combined[id.item()] for id in test_tags]
        predictions = [id2label_combined[id.item()] for id in test_preds]
        final_f1 = f1_score(y_true=tags, y_pred=predictions, average='micro')
        final_macro_f1 = f1_score(y_true=tags, y_pred=predictions, average='macro')
        final_acc = accuracy_score(y_true=tags, y_pred=predictions)
        print(final_f1)
        print(final_macro_f1)
        print(final_acc)
        print(classification_report([tags], [predictions]))
        return final_f1