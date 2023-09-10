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
from torch import nn
from TorchCRF import CRF
from transformers import BertModel, BertPreTrainedModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 2
EPOCHS = 3
0
LEARNING_RATE = 4e-05
MAX_NORM = 10
#tokenizer = BertTokenizer.from_pretrained('bert_model-base-uncased')
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

PARAM_TRAIN = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 10}
PARAM_TEST = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 10
              }


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


class BertBiLSTMCRF(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    def __init__(self, config):
        super().__init__(config)
        self.tag_count = config.num_labels
        self.bert_model = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bi_lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, dropout=0.2, batch_first=True, bidirectional=True)
        self.linear_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if return_dict is not None:
            return_dict = return_dict
        else:
            self.config.use_return_dict

        result = self.bert_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence = self.dropout(result[0])
        lstm_output, hc = self.bi_lstm(sequence)
        logits = self.linear_layer(sequence)
        log_likelihood = self.crf(logits, labels, reduction='mean')
        tags = self.crf.decode(logits)
        loss = 0 - log_likelihood
        tags = torch.Tensor(tags)
        if not return_dict:
            output = (tags,) + result[2:]
            return ((loss,) + output) if loss is not None else output
        return loss, tags


class BertCRF(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.tag_count = config.num_labels
        self.bert_model = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.pos_emb = torch.nn.Embedding(config.hidden_size, 1)
        self.dep_emb = torch.nn.Embedding(config.hidden_size, 1)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pos_tags=None,
        dependencies=None,
    ):
        if return_dict is not None:
            return_dict = return_dict
        else:
            self.config.use_return_dict

        result = self.bert_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence = result[0]
        sequence = self.dropout(sequence)
        # get additional feature embeddings and concatenate them with the bert contextualized embeddings
        if pos_tags is not None:
            pos_emb = self.pos_emb(pos_tags)
            pos_emb = self.dropout(pos_emb)
            sequence = torch.cat((sequence, pos_emb), dim=-1)
        if dependencies is not None:
            dep_emb = self.dep_emb(dependencies)
            dep_emb = self.dropout(dep_emb)
            sequence = torch.cat((sequence, dep_emb), dim=-1)

        # Modified concatenate contextualized embeddings from BERT and your categorical embedding

        logits = self.linear_layer(sequence)
        log_likelihood = self.crf(logits, labels, reduction='mean')
        tags = self.crf.decode(logits)
        loss = 0 - log_likelihood
        tags = torch.Tensor(tags)
        if not return_dict:
            output = (tags,) + result[2:]
            return ((loss,) + output) if loss is not None else output
        return loss, tags


class BertExtended:
    def __init__(self, name, filter_dataset=False):
        self.data = prep.Dataloader(train_path='train.json', test_path='test.json', filter_dataset=filter_dataset, bio=True)
        self.train_dataset = prep.BertPrepper(sentences=self.data.train_sentences, tags=self.data.train_tags, pos_tags=self.data.train_pos_tags, dependencies=self.data.train_dependencies, unique_tags=self.data.unique_tags, max_len=128, all_pos=self.data.all_pos, all_deps=self.data.all_deps)
        self.test_dataset = prep.BertPrepper(sentences=self.data.test_sentences, tags=self.data.test_tags, pos_tags=self.data.test_pos_tags, dependencies=self.data.test_dependencies, unique_tags=self.data.unique_tags, max_len=128, all_pos=self.data.all_pos, all_deps=self.data.all_deps)
        self.train_loaded = DataLoader(self.train_dataset, **PARAM_TRAIN)
        self.test_loaded = DataLoader(self.test_dataset, **PARAM_TEST)
        self.model = self.make_model(name=name)
        self.epoch_stop = EarlyStopping(patience=5)
        self.early_stopping = EarlyStopping(patience=5)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            print('Training of epoch: ', epoch + 1)
            epoch_loss = self.train(epoch)
            #if self.epoch_stop.step(epoch_loss):
             #   break
        self.final_results = self.test()
        if not os.path.exists('models'):
            os.mkdir('models')
        self.model.save_pretrained(os.path.join('models', 'bert_pretrained.h5'))

    def make_model(self, name):
        model = None
        if name == 'bert_crf':
            model = BertCRF.from_pretrained('bert-base-cased', num_labels=len({**self.train_dataset.id2label, **self.test_dataset.id2label}), id2label={**self.train_dataset.id2label,**self.test_dataset.id2label}, label2id={**self.train_dataset.label2id,**self.test_dataset.label2id}).to(DEVICE)
            #model = BertCRF.from_pretrained('dmis-lab/biobert-base-cased-v1.2', num_labels=len({**self.train_dataset.id2label, **self.test_dataset.id2label}), id2label={**self.train_dataset.id2label,**self.test_dataset.id2label}, label2id={**self.train_dataset.label2id,**self.test_dataset.label2id}).to(DEVICE)
        elif name == 'bert_bilstm_crf':
            model = BertBiLSTMCRF.from_pretrained('bert-base-cased', num_labels=len(
                {**self.train_dataset.id2label, **self.test_dataset.id2label}),
                                            id2label={**self.train_dataset.id2label, **self.test_dataset.id2label},
                                            label2id={**self.train_dataset.label2id, **self.test_dataset.label2id}).to(
                DEVICE)
        return model

    def get_f1_score(self, targets, logits, mask, predictions, tags, full_f1):
        flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
        active_logits = logits.view(-1).to(DEVICE)  # shape (batch_size * seq_len, tag_count)
        active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        preds = torch.masked_select(active_logits, active_accuracy.to(DEVICE))
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
            ids = batch['ids'].to(DEVICE, dtype=torch.long)
            mask = batch['mask'].to(DEVICE, dtype=torch.long)
            targets = batch['targets'].to(DEVICE, dtype=torch.long)
            pos_embs = batch['pos_embs'].to(DEVICE, dtype=torch.long)
            dep_embs = batch['dep_embs'].to(DEVICE, dtype=torch.long)
            #pos_embs = batch[]
            result = self.model(input_ids=ids, attention_mask=mask, labels=targets)
            #result = self.model(input_ids=ids, attention_mask=mask, labels=targets, pos_tags, dependencies)
            loss = result[0]
            logits = result[1]

            train_loss += loss.item()
            train_step_count += 1
            if i % 100 == 0:
                loss_step = train_loss/train_step_count
                print('Training: Loss per 100 steps: ', loss_step)
                #if self.early_stopping.step(loss_step):
                 #   print('stop mid epoch')
                  #  break
            # get the f1-score
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            f1_train, predictions, tags = self.get_f1_score(targets=targets, logits=logits, mask=mask, predictions=predictions, tags=tags, full_f1=f1_train)
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=MAX_NORM)
            # backward pass
            self.optimizer.zero_grad()
            gc.collect()
            #if torch.cuda.is_available():
             #   torch.cuda.empty_cache()
              #  print(torch.cuda.memory_summary(DEVICE=DEVICE))
            loss.backward()
            self.optimizer.step()
        final_loss = train_loss/train_step_count
        print(f"loss of epoch: {final_loss}")
        f1_train = f1_train/train_step_count
        print(f"f1-score of epoch: {f1_train}")
        return final_loss

    def test(self):
        self.model.eval()
        test_loss = 0
        f1_val = 0
        nb_eval_steps = 0
        test_preds, test_tags = [], []

        with torch.no_grad():
            for idx, batch in enumerate(self.test_loaded):
                ids = batch['ids'].to(DEVICE, dtype=torch.long)
                mask = batch['mask'].to(DEVICE, dtype=torch.long)
                targets = batch['targets'].to(DEVICE, dtype=torch.long)
                outputs = self.model(input_ids=ids, attention_mask=mask, labels=targets)
                loss, eval_logits = outputs[0], outputs[1]
                test_loss += loss.item()
                nb_eval_steps += 1
                if idx % 100 == 0:
                    loss_step = test_loss / nb_eval_steps
                    print('Validation: Loss per 100 steps: ', loss_step)
                f1_val, test_preds, test_tags = self.get_f1_score(targets=targets, logits=eval_logits, mask=mask, predictions=test_preds, tags = test_tags, full_f1=f1_val)
        id2label_combined = {**self.train_dataset.id2label,  **self.test_dataset.id2label}
        tags = [id2label_combined[id.item()] for id in test_tags]
        predictions = [id2label_combined[id.item()] for id in test_preds]
        test_loss = test_loss / nb_eval_steps
        final_f1 = f1_val / nb_eval_steps
        print('Test Losss: ',test_loss)
        print('Test Accuracy: ', final_f1)
        report = classification_report([tags], [predictions])
        print(report)

        return tags, predictions, final_f1, report
