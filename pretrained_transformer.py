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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = 'cuda' if cuda.is_available() else 'cpu'
#https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 3e-05
MAX_NORM = 10
NUM_WORKERS = 1
TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': NUM_WORKERS}
VALID_PARAMS = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': NUM_WORKERS}
TEST_PARAMS = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': NUM_WORKERS}


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=5):
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
        change = self.check(loss, min_delta=0.01)
        if change:
            self.num_bad_epochs = 0
            self.best = loss
        else:
            self.num_bad_epochs += 1
        print('count of bad epochs', self.num_bad_epochs)
        if self.num_bad_epochs >= self.patience:
            a=0
            #print('terminating because of early stopping!')
            #return True
        return False

    def check(self, loss, min_delta):
        change = False
        if (self.best - loss) > min_delta:
            self.is_better = True
            change = True
        else:
            self.is_better = False
            change = False
        return change


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
        self.epoch_stop = EarlyStopping(patience=5)
        self.early_stopping = EarlyStopping(patience=5)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            print(f"Training epoch: {epoch + 1}")
            epoch_loss = self.train(epoch)
            if self.epoch_stop.step(epoch_loss):
                print('tis enough')
                #break
        self.validate()
        self.final_f1 = self.test()
        if not os.path.exists('models'):
            os.mkdir('models')
        self.model.save_pretrained(os.path.join('models', 'bert_pretrained.h5'))

    def get_f1_score(self, targets, logits, mask, predictions, tags, full_f1):
        flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)        # shape (batch_size * seq_len, tag_count)
        flattened_predictions = torch.argmax(input=active_logits, dim=1)  # prev axis = 1 shape (batch_size * seq_len,)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        preds = torch.masked_select(flattened_predictions, active_accuracy)
        predictions.extend(preds)
        tags.extend(targets)
        cur_f1 = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='micro')
        full_f1 += cur_f1
        return full_f1, predictions, tags

    def train(self, epoch):
        # init variables
        train_loss = 0
        f1_train = 0
        train_step_count = 0
        predictions = []
        tags = []
        # start training
        self.model.train()
        for i, batch in enumerate(self.train_loaded):
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)
            out = self.model(input_ids=ids, attention_mask=mask, labels=targets)
            loss = out.loss
            logits = out.logits
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
            f1_train, predictions, tags = self.get_f1_score(targets=targets, logits=logits, mask=mask, predictions=predictions, tags=tags, full_f1=f1_train)
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=MAX_NORM)
            # backward pass
            self.optimizer.zero_grad()
            gc.collect()
            loss.backward()
            self.optimizer.step()
        final_loss = train_loss/train_step_count
        print(f"loss of epoch: {final_loss}")
        f1_train = f1_train/train_step_count
        print(f"f1-score of epoch: {f1_train}")
        return final_loss

    def validate(self):
        self.model.eval()
        eval_loss = 0
        f1_val = 0
        nb_eval_steps = 0
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
                nb_eval_steps += 1
                if idx % 100 == 0:
                    loss_step = eval_loss / nb_eval_steps
                    print('Validation: Loss per 100 steps: ', loss_step)

                f1_val, val_preds, val_tags = self.get_f1_score(targets=targets, logits=eval_logits, mask=mask, predictions=val_preds, tags=val_tags, full_f1=f1_val)
        id2label_combined = {**self.train_dataset.id2label, **self.valid_dataset.id2label, **self.test_dataset.id2label}
        tags = [id2label_combined[id.item()] for id in val_tags]
        predictions = [id2label_combined[id.item()] for id in val_preds]
        eval_loss = eval_loss / nb_eval_steps
        final_f1 = f1_val / nb_eval_steps
        print('Validation Loss: ', eval_loss)
        print('Validation F1-score: ', final_f1)
        print(classification_report([tags], [predictions]))

        return final_f1

    def test(self):
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
            f1_test, test_preds, test_tags = self.get_f1_score(targets=targets, logits=logits, mask=mask,
                                                            predictions=test_preds, tags=test_tags, full_f1=f1_test)
            word_level_predictions = []
            #for pair in wp_preds:
             #   if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                    # skip prediction
              #      continue
               # else:
                #    word_level_predictions.append(pair[1])
            samples += 1

        # we join tokens, if they are not special ones
            #str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
            #print(str_rep)
            #print(word_level_predictions)
        final_f1 = f1_test / samples
        print(final_f1)
        return final_f1