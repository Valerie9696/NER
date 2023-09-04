import os.path
import string

import gensim
import json
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
import keras
import numpy as np
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification


class Dataloader:
    def __init__(self, train_path, test_path, filter_dataset=False, bio=False):
        self.train_data = self.load_data(train_path)
        self.test_data = self.load_data(test_path)
        self.train_sentences, self.train_tags = self.make_tags(self.train_data, filter=filter_dataset, bio=bio)
        self.test_sentences, self.test_tags = self.make_tags(self.test_data, filter=filter_dataset, bio=bio)
        self.unique_tags = set(list(self.get_unique_words(sentences=self.train_tags))+list(self.get_unique_words(self.test_tags)))
        self.lookup_table = self.make_lookup()
        self.vocabulary = self.get_vocabulary()
        self.lookup_layer = keras.layers.StringLookup(vocabulary=self.vocabulary)

    def load_data(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)
            f.close()
        return data

    def get_unique_words(self, sentences):
        words = []
        for sentence in sentences:
            words.extend(set(sentence))
        return set(words)

    def make_tags(self, data, filter=False, bio=False):
        abstracts = []
        all_tags = []
        counter = 0
        for abstract in data:
            sentences = []
            tags = []
            tuples = []
            for sentence in abstract['sentences']:
                entities = sentence['entities']
                tagged = ['O'] * len(sentence['words'])
                for entity in entities:
                    sentences.append(sentence['words'])
                    if len(entity['words']) > 1:
                        indices = range(entity['start_pos'], entity['end_pos'])
                        for index in indices:
                            if tagged[index] != 'O':
                                print('overlap')
                                counter += 1
                            if bio:
                                if index == indices[0]:
                                    tagged[index] = 'B-' + entity['label']
                                else:
                                    tagged[index] = 'I-' + entity['label']
                            else:
                                tagged[index] = entity['label']
                    elif len(entity['words']) == 1:
                        if bio:
                            tagged[entity['start_pos']] = 'B-' + entity['label']
                        else:
                            tagged[entity['start_pos']] = entity['label']
                    tags.append(tagged)
                    tuples.append(tuple(zip(sentence['words'], tagged)))
            abstracts = abstracts + sentences
            all_tags = all_tags + tags
            # dataset augmentation by removing punctuation and single character words
        if filter:
            for i in range(0, len(abstracts)):
                sentence = abstracts[i]
                tags = all_tags[i]
                sentence_dupe = sentence.copy()
                del_indices = []
                for j in range(0, len(sentence)):
                    word = sentence_dupe[j]
                    # remove single character words
                    if len(word) < 2 or word.isnumeric():
                        del_indices.append(j)
                    # remove words containing punctuation
                    elif len(word) > 1:
                        for char in word:
                            if char in string.punctuation:
                                del_indices.append(j)
                for idx in sorted(del_indices, reverse=True):
                    del sentence[idx]
                    del tags[idx]

        return abstracts, all_tags

    def make_lookup(self):
        unique_tags = self.get_unique_words(self.train_tags)
        table = dict(zip(range(0, len(unique_tags) + 1), unique_tags))
        return table
    #vocab has only 3148 words, but index 3204 is tried to be looked up
    def get_vocabulary(self):
        train_vocab = self.get_unique_words(self.train_sentences)
        test_vocab = self.get_unique_words(self.test_sentences)
        vocab = set(list(train_vocab) + list(test_vocab))
        table_arr = np.array(list(map(str.lower, vocab)))
        counter = Counter(table_arr)
        vocab_size = len(counter)
        vocabulary = [token for token, count in counter.most_common(vocab_size)]
        return vocabulary


class LSTMPrepper:
    def __init__(self):
        self.dl = Dataloader(train_path='train.json', test_path='test.json')
        #self.x_train_padded, self.x_test_padded, self.y_train_padded, self.y_test_padded, self.max_len = self.pad(train=self.dl.train_sentences, test=self.dl.test_sentences)
        self.tokenizer, self.embedding, self.x_train_padded, self.y_train_padded, self.x_test_padded, self.y_test_padded, self.max_len = self.tokenize()
        self.y_train = keras.utils.to_categorical(self.y_train_padded)
        self.y_test = keras.utils.to_categorical(self.y_test_padded)

    def find_max_sublist(self, data):
        max_list = max(data, key=len)
        max_len = max(map(len, data))
        return max_len

    def tokenize(self):
        sentences = list(self.dl.train_sentences) + list(self.dl.test_sentences)
        tokenizer = Tokenizer()
        if not os.path.exists('Embeddings'):
            os.mkdir('Embeddings')
        if not os.path.isfile(os.path.join('Embeddings', 'w2v.word2vec')):
            w2v_model = gensim.models.Word2Vec(sentences=sentences, min_count=5, window=5, sg=1)
            w2v_model.save(os.path.join('Embeddings', 'w2v.word2vec'))
        else:
            w2v_model = gensim.models.Word2Vec.load(os.path.join('Embeddings', 'w2v.word2vec'))
        GLOVE_DIM = 181
        tokenizer.fit_on_texts(sentences)
        train_tokenized = tokenizer.texts_to_sequences(self.dl.train_sentences)
        test_tokenized = tokenizer.texts_to_sequences(self.dl.test_sentences)
        train_max = self.find_max_sublist(self.dl.train_sentences)
        test_max = self.find_max_sublist(self.dl.train_sentences)
        pad_len = max(train_max, test_max)
        t_index = {t: j for j, t in enumerate(self.dl.unique_tags)}
        y_train = [[t_index[w] for w in t] for t in self.dl.train_tags]
        y_test = [[t_index[w] for w in t] for t in self.dl.test_tags]
        y_train_padded = pad_sequences(maxlen=pad_len, padding='post', sequences=y_train)
        y_test_padded = pad_sequences(maxlen=pad_len, padding='post', sequences=y_test)

        x_train_padded = pad_sequences(maxlen=pad_len, padding='post', sequences=train_tokenized)
        x_test_padded = pad_sequences(maxlen=pad_len, padding='post', sequences=test_tokenized)
        word_count = len(tokenizer.word_index)
        emb_matrix = np.zeros((word_count + 1, GLOVE_DIM))
        word_items = tokenizer.word_index.items()
        for w, i in word_items:
            # The word_index contains a token for all words of the training data, so we need to limit that
            if i < word_count:
                try:
                    vect = w2v_model.wv.get_vector(w)
                    emb_matrix[i] = vect
                except:
                    pass
                # Check if the word from the training data occurs in the GloVe word embeddings
                # Otherwise the vector is kept with only zeros
            else:
                break
        a=0
        return tokenizer, emb_matrix, x_train_padded, y_train_padded, x_test_padded, y_test_padded, pad_len
    def pad(self, train, test):
        train_max = self.find_max_sublist(train)
        test_max = self.find_max_sublist(test)
        pad_len = max(train_max, test_max)
        words = set(list(self.dl.get_unique_words(train)) + list(self.dl.get_unique_words(test)))
        tags = self.dl.unique_tags
        w_index = {w: i for i, w in enumerate(words)}
        t_index = {t: j for j, t in enumerate(tags)}
        x_train = [[w_index[w] for w in s] for s in self.dl.train_sentences]
        x_test = [[w_index[w] for w in s] for s in self.dl.test_sentences]

        y_train = [[t_index[w] for w in t] for t in self.dl.train_tags]
        y_test = [[t_index[w] for w in t] for t in self.dl.test_tags]
        y_train_padded = pad_sequences(maxlen=pad_len, padding='post', sequences=y_train)
        y_test_padded = pad_sequences(maxlen=pad_len, padding='post', sequences=y_test)

        x_train_padded = pad_sequences(maxlen=pad_len, padding='post', sequences=x_train)
        x_test_padded = pad_sequences(maxlen=pad_len, padding='post', sequences=x_test)
        return x_train_padded, x_test_padded, y_train_padded, y_test_padded, pad_len


class BertPrepper(Dataset):
    def __init__(self, sentences, tags, unique_tags, max_len):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.len = len(sentences)
        self.label2id = {k: v for v, k in enumerate(unique_tags)}
        self.id2label = {v: k for v, k in enumerate(unique_tags)}

    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.sentences[index]
        word_labels = self.tags[index]
        tokenized_sentence, labels = self.tokenize_and_preserve_labels(sentence=sentence, tags=word_labels)

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
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def tokenize_and_preserve_labels(self, sentence=None, tags=None):
        tokenized_sentence = []
        tokenized_tags = []
        # sentence = sentence.strip()
        for word, tag in zip(sentence, tags):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized = self.tokenizer.tokenize(word)
            sub_words = len(tokenized)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized)

            # Add the same label to the new list of labels `n_subwords` times
            tokenized_tags.extend([tag] * sub_words)

        return tokenized_sentence, tokenized_tags

    def __len__(self):
        return self.len



#lstm_prep = LSTMPrepper()