import os
import gc
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
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
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EPOCHS = 15#8
LEARNING_RATE = 3e-05 # 0.00005#4e-05
MAX_NORM = 1
NUM_WORKERS = 8
SHUFFLE = False
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

PARAM_TRAIN = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': SHUFFLE, 'num_workers': NUM_WORKERS}
PARAM_VALID = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': SHUFFLE, 'num_workers': NUM_WORKERS}
PARAM_TEST = {'batch_size': TEST_BATCH_SIZE, 'shuffle': SHUFFLE, 'num_workers': NUM_WORKERS}


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
        logits = self.linear_layer(lstm_output)
        loss = None
        if labels is not None:
            mask = []
            for i in attention_mask:
                m = []
                for x in i:
                    if x == 0:
                        m.append(False)
                    else:
                        m.append(True)
                mask.append(m)
            mask = torch.tensor(np.array(mask)).to(DEVICE)
            log_likelihood = self.crf(logits, labels, mask=mask, reduction='mean')
            tags = self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)
        if not return_dict:
            output = (tags,) + result[2:]
            return ((loss,) + output) if loss is not None else output
        return loss, tags


class BertCRF(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, has_features=False):
        super().__init__(config)
        self.tag_count = config.num_labels
        self.bert_model = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if has_features:
            self.linear_layer = nn.Linear(config.hidden_size + 2, config.num_labels)
            self.pos_emb = torch.nn.Embedding(config.hidden_size, 1)
            self.dep_emb = torch.nn.Embedding(config.hidden_size, 1)
        else:
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
        pos_tags=None,
        dependencies=None,
        epochs=None,
    ):
        if return_dict is not None:
            return_dict = return_dict
        else:
            self.config.use_return_dict

        out = self.bert_model(
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
        # fine tune bert for 4 epochs, only train crf thereafter
        if epochs is not None:
            if epochs > 10:
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            else:
                for param in self.bert_model.parameters():
                    param.requires_grad = True
        sequence = out[0]
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
        logits = self.linear_layer(sequence)
        loss = None
        if labels is not None:
            mask = []
            for i in attention_mask:
                m = []
                for x in i:
                    if x == 0:
                        m.append(False)
                    else:
                        m.append(True)
                mask.append(m)
            mask = torch.tensor(np.array(mask)).to(DEVICE)
            log_likelihood = self.crf(emissions=logits, tags=labels, mask=mask, reduction='mean') # marker mask added
            tags = self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)
        if not return_dict:
            output = (tags,) + out[2:]
            return ((loss,) + output) if loss is not None else output
        return loss, tags


class BertExtended:
    def __init__(self, name, filter_dataset=False, with_features=False):
        self.data = prep.Dataloader(train_path='train.json', test_path='test.json', filter_dataset=filter_dataset,
                                    bio=True)
        self.train_dataset = prep.BertPrepper(sentences=self.data.train_sentences, tags=self.data.train_tags,
                                              pos_tags=self.data.train_pos_tags,
                                              dependencies=self.data.train_dependencies,
                                              unique_tags=self.data.unique_tags, max_len=128, all_pos=self.data.all_pos,
                                              all_deps=self.data.all_deps)
        self.valid_dataset = prep.BertPrepper(sentences=self.data.valid_sentences, tags=self.data.valid_tags,
                                              pos_tags=self.data.valid_pos_tags,
                                              dependencies=self.data.valid_dependencies,
                                              unique_tags=self.data.unique_tags, max_len=128, all_pos=self.data.all_pos,
                                              all_deps=self.data.all_deps)
        self.test_dataset = prep.BertPrepper(sentences=self.data.test_sentences, tags=self.data.test_tags,
                                             pos_tags=self.data.test_pos_tags, dependencies=self.data.test_dependencies,
                                             unique_tags=self.data.unique_tags, max_len=128, all_pos=self.data.all_pos,
                                             all_deps=self.data.all_deps)
        self.train_loaded = DataLoader(self.train_dataset, **PARAM_TRAIN)
        self.valid_loaded = DataLoader(self.valid_dataset, **PARAM_VALID)
        self.test_loaded = DataLoader(self.test_dataset, **PARAM_TEST)
        self.with_features = with_features
        self.model = self.make_model(name=name, with_features=with_features)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            print('Training of epoch: ', epoch + 1)
            epoch_loss = self.train(epoch)
        self.validate()
        self.final_f1 = self.test()
        if not os.path.exists('models'):
            os.mkdir('models')
        self.model.save_pretrained(os.path.join('models', 'bert_crf.h5'))

    def make_model(self, name, with_features=False):
        model = None
        if name == 'bert_crf':
            model = BertCRF.from_pretrained('bert-base-uncased', num_labels=len(self.train_dataset.id2label),
                                            id2label=self.train_dataset.id2label, label2id=self.train_dataset.label2id,
                                            has_features=with_features).to(DEVICE)
            #model = BertCRF.from_pretrained('dmis-lab/biobert-base-cased-v1.2', num_labels=len({**self.train_dataset.id2label, **self.test_dataset.id2label}), id2label={**self.train_dataset.id2label,**self.test_dataset.id2label}, label2id={**self.train_dataset.label2id,**self.test_dataset.label2id}).to(DEVICE)
        elif name == 'bert_lstm_crf':
            model = BertBiLSTMCRF.from_pretrained('bert-base-uncased', num_labels=len(self.train_dataset.id2label),
                                                  id2label=self.train_dataset.id2label, label2id=self.train_dataset.label2id)
        return model


    def get_metric(self, targets, logits, mask, predictions, tags, full_val, metric='accuracy'):
        flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
        active_logits = logits.view(-1).to(DEVICE)  # shape (batch_size * seq_len, tag_count)
        active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        preds = torch.masked_select(active_logits, active_accuracy.to(DEVICE))
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
        # init variables
        train_loss = 0
        acc_train = 0
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

            if self.with_features:
                out = self.model(input_ids=ids, attention_mask=mask, labels=targets, pos_tags=pos_embs, dependencies=dep_embs, epochs=epoch)
            else:
                out = self.model(input_ids=ids, attention_mask=mask, labels=targets, epochs=epoch)
            loss = out[0]
            logits = out[1]
            train_loss += loss.item()
            train_step_count += 1
            if i % 100 == 0:
                loss_step = train_loss/train_step_count
                print('Training: Loss per 100 steps: ', loss_step)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            acc_train, predictions, tags = self.get_metric(targets=targets, logits=logits, mask=mask,
                                                           predictions=predictions, tags=tags, full_val=acc_train,
                                                           metric='accuracy')
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=MAX_NORM)
            # backward pass
            self.optimizer.zero_grad()
            gc.collect()
            loss.backward()
            self.optimizer.step()
        final_loss = train_loss/train_step_count
        print('Training: Loss of epoch: ', final_loss)
        full_acc = acc_train/train_step_count
        print('Accuracy of epoch: ', full_acc)
        return final_loss

    def validate(self):
        self.model.eval()
        test_loss = 0
        acc_val = 0
        val_steps = 0
        val_preds, val_tags = [], []
        with torch.no_grad():
            for idx, batch in enumerate(self.valid_loaded):
                ids = batch['ids'].to(DEVICE, dtype=torch.long)
                mask = batch['mask'].to(DEVICE, dtype=torch.long)
                targets = batch['targets'].to(DEVICE, dtype=torch.long)
                pos_embs = batch['pos_embs'].to(DEVICE, dtype=torch.long)
                dep_embs = batch['dep_embs'].to(DEVICE, dtype=torch.long)
                if self.with_features:
                    out = self.model(input_ids=ids, attention_mask=mask, labels=targets, pos_tags=pos_embs,
                                     dependencies=dep_embs)
                else:
                    out = self.model(input_ids=ids, attention_mask=mask, labels=targets)
                loss, eval_logits = out[0], out[1]
                test_loss += loss.item()
                val_steps += 1
                if idx % 100 == 0:
                    loss_step = test_loss / val_steps
                    print('Validation: Loss per 100 steps: ', loss_step)
                acc_val, val_preds, val_tags = self.get_metric(targets=targets, logits=eval_logits, mask=mask,
                                                               predictions=val_preds, tags=val_tags,
                                                               full_val=acc_val, metric='accuracy')
        id2l = self.train_dataset.id2label
        tags = [id2l[id.item()] for id in val_tags]
        predictions = [id2l[id.item()] for id in val_preds]
        test_loss = test_loss / val_steps
        final_acc = acc_val / val_steps
        print('Validation Loss: ', test_loss)
        print('Validation Accuracy: ', final_acc)
        return tags, predictions, final_acc

    def test(self):
        f1_test = 0
        acc_test = 0
        samples = 0
        test_preds = []
        test_tags = []
        all_tokens = None
        conv_tokens = None
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(self.test_loaded):
                ids = batch['ids'].to(DEVICE, dtype=torch.long)
                mask = batch['mask'].to(DEVICE, dtype=torch.long)
                targets = batch['targets'].to(DEVICE, dtype=torch.long)
                pos_embs = batch['pos_embs'].to(DEVICE, dtype=torch.long)
                dep_embs = batch['dep_embs'].to(DEVICE, dtype=torch.long)
                if self.with_features:
                    out = self.model(input_ids=ids, attention_mask=mask, labels=targets, pos_tags=pos_embs,
                                     dependencies=dep_embs)
                    logits = out[1]
                    samples += 1
                    # active_logits = logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)
                    #flattened_predictions = torch.argmax(logits, axis=1)
                    # tokens = self.test_dataset.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
                    active_logits = logits.view(-1).to(DEVICE)  # shape (batch_size * seq_len, tag_count)
                    active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
                    preds = torch.masked_select(active_logits, active_accuracy.to(DEVICE))
                    id2l = self.train_dataset.id2label
                    token_predictions = []
                    #for i in flattened_predictions.cpu().numpy():
                    for i in preds.cpu().numpy():
                        try:
                            token_predictions.append(id2l[i])
                        except:
                            print('booooooo')
                            print(i)
                            print(id2l)
                    # token_predictions = [id2l[i] for i in flattened_predictions.cpu().numpy()]
                    # wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)
                    f1_test, test_preds, test_tags = self.get_f1_score(targets=targets, logits=logits, mask=mask,
                                                                       predictions=test_preds, tags=test_tags,
                                                                       full_f1=f1_test)
                    word_level_predictions = []
                    # for pair in wp_preds:
                    #   if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                    # skip prediction
                    #      continue
                    # else:
                    #    word_level_predictions.append(pair[1])
                else:
                    out = self.model(input_ids=ids, attention_mask=mask)
                    logits = out[0]
                    active_logits = logits.view(-1).to(DEVICE)  # shape (batch_size * seq_len, tag_count)
                    active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
                    preds = torch.masked_select(active_logits, active_accuracy.to(DEVICE))
                    samples += 1
                    # active_logits = logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)
                    #flattened_predictions = torch.argmax(logits, axis=1)
                    # tokens = self.test_dataset.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
                    id2l = self.train_dataset.id2label
                    token_predictions = []
                    for i in preds.cpu().numpy():
                        try:
                            token_predictions.append(id2l[i])
                        except:
                            print('booooooo')
                            print(i)
                            print(id2l)
                    # token_predictions = [id2l[i] for i in flattened_predictions.cpu().numpy()]
                    # wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)
                    acc_test, test_preds, test_tags = self.get_metric(targets=targets, logits=logits, mask=mask,
                                                                      predictions=test_preds, tags=test_tags,
                                                                      full_val=acc_test, metric='accuracy')

                    for tokens in ids.cpu().numpy():
                        if conv_tokens is None:
                            conv_tokens = self.train_dataset.tokenizer.convert_ids_to_tokens(
                                tokens.squeeze().tolist())
                        else:
                            conv_tokens.extend(self.train_dataset.tokenizer.convert_ids_to_tokens(
                                tokens.squeeze().tolist()))
            if False:
                tokens_tags = list(zip(conv_tokens, test_preds))
                word_level_predictions = []
                for pair in tokens_tags:
                    if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                        continue
                    else:
                        word_level_predictions.append(pair[1])
                str_rep = " ".join([t[0] for t in tokens_tags if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
                print(str_rep)
                print(word_level_predictions)

            # we join tokens, if they are not special ones
            # str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
            # print(str_rep)
            # print(word_level_predictions)
        # we join tokens, if they are not special ones
        # str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
        # print(str_rep)
        # print(word_level_predictions)
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
