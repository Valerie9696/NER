import json
#from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from keras.utils import to_categorical
import keras
import numpy as np
from collections import Counter


class Dataloader:
    def __init__(self, train_path, test_path):
        self.train_data = self.load_data(train_path)
        self.test_data = self.load_data(test_path)
        self.train_sentences, self.train_tags = self.make_tags(self.train_data)
        self.test_sentences, self.test_tags = self.make_tags(self.test_data)
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

    def make_tags(self, data):
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
                        for index in range(entity['start_pos'], entity['end_pos']):
                            if tagged[index] != 'O':
                                print('overlap')        # todo: filter overlaps
                                counter += 1
                            tagged[index] = entity['label']
                    elif len(entity['words']) == 1:
                        tagged[entity['start_pos']] = entity['label']
                    tags.append(tagged)
                    tuples.append(tuple(zip(sentence['words'], tagged)))
            abstracts = abstracts + sentences
            all_tags = all_tags + tags
        print(counter)
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
        self.x_train_padded, self.x_test_padded, self.y_train_padded, self.y_test_padded, self.max_len = self.pad(train=self.dl.train_sentences, test=self.dl.test_sentences)
        self.y_train = keras.utils.to_categorical(self.y_train_padded)
        self.y_test = keras.utils.to_categorical(self.y_test_padded)

    def find_max_sublist(self, data):
        max_list = max(data, key=len)
        max_len = max(map(len, data))
        return max_len

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

lstm_prep = LSTMPrepper()