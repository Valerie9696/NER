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
import spacy


class Dataloader:
    def __init__(self, train_path, test_path, filter_dataset=False, bio=False):
        self.train_data, self.validation_data = self.split_train_valid(self.load_data(train_path))
        self.test_data = self.load_data(test_path)
        self.nlp = spacy.load('en_core_web_sm')
        self.train_sentences, self.train_tags, self.train_pos_tags, self.train_dependencies = self.make_tags(self.train_data, filter=filter_dataset, bio=bio)
        self.valid_sentences, self.valid_tags, self.valid_pos_tags, self.valid_dependencies = self.make_tags(self.validation_data, filter=filter_dataset, bio=bio)
        self.test_sentences, self.test_tags, self.test_pos_tags, self.test_dependencies = self.make_tags(self.test_data, filter=filter_dataset, bio=bio)
        self.unique_tags = set(list(self.get_unique_words(sentences=self.train_tags))+list(self.get_unique_words(self.valid_tags))+list(self.get_unique_words(self.test_tags)))
        self.all_pos = self.get_unique_words(self.train_pos_tags) + self.get_unique_words(self.valid_pos_tags) + self.get_unique_words(self.test_pos_tags)# + ['det'] + ['ROOT', 'compound'])
        self.all_deps = self.get_unique_words(self.train_dependencies) + self.get_unique_words(self.valid_dependencies) + self.get_unique_words(self.test_dependencies)
        self.lookup_table = self.make_lookup()
        self.vocabulary = self.get_vocabulary()
        self.lookup_layer = keras.layers.StringLookup(vocabulary=self.vocabulary)


    def load_data(self, path):
        """
        Load json file from a given path.
        :param path: path to the file
        :return: content of file
        """
        with open(path, 'rb') as f:
            data = json.load(f)
            f.close()
        return data

    def get_unique_words(self, sentences):
        """
        Find all unique words in a list of sentences in order to build up a vocabulary.
        :param sentences: list of sentences
        :return: unique words from those sentences
        """
        words = []
        for sentence in sentences:
            #words.extend(np.unique(sentence))
            for word in sentence:
                if word not in words:
                    words.append(word)
        #final = np.unique(words)

        return words#np.unique(words)

    def make_pos_dep_tags(self, sentence):
        """
        Generate a list of POS tags and dependencies for a given sentence.
        :param sentence: input sentence
        :return: list of POS tags, list of dependencies
        """
        joined = ' '.join(word for word in sentence)
        spacy_sentence = self.nlp(joined)  # put sentence in the nlp pipeline of spacy
        pos_tags = []
        dependencies = []
        stop_words = []
        for tagged_word in spacy_sentence:  # add tag to list of pos tags for each word of the sentence
            pos_tags.append(tagged_word.pos_)
            dependencies.append(tagged_word.dep_)
            stop_words.append(tagged_word.is_stop)
        #print(len(sentence), len(pos_tags))
        return pos_tags, dependencies

    def make_tags(self, data, filter=False, bio=False):
        abstracts = []
        all_tags = []
        all_pos_tags = []
        all_dependencies = []
        counter = 0
        for abstract in data:
            sentences = []
            tags = []
            tuples = []
            pos_tags = []
            dependencies = []
            for sentence in abstract['sentences']:
                pos, dep = self.make_pos_dep_tags(sentence['words'])
                pos_tags.append(pos)
                dependencies.append(dep)
                entities = sentence['entities']
                sentences.append(sentence['words'])
                tagged = ['O'] * len(sentence['words'])
                for entity in entities:
                    if len(entity['words']) > 1:
                        indices = range(entity['start_pos'], entity['end_pos'])
                        # print('Länge: ', entity['end_pos']-entity['start_pos'])
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
            all_pos_tags = all_pos_tags + pos_tags
            all_dependencies = all_dependencies + dependencies
            # dataset augmentation by removing punctuation and single character words
        if filter:
            f_abstracts, f_all_tags, f_all_pos, f_all_dep = [], [], [], []
            for i in range(0, len(abstracts)):
                sentence = abstracts[i]
                tags = all_tags[i]
                pos = all_pos_tags[i]
                dep = all_dependencies[i]
                del_indices = []
                #### check first if the tag is an entity only delete if not
                for j in range(0, len(sentence)):
                    word = sentence[j]
                    tag = tags[j]
                    # remove single character words
                    if tag == 'O':
                        if len(word) < 2 or word.isnumeric():
                            del_indices.append(j)
                        # remove words containing punctuation
                        elif len(word) > 1:
                            for char in word:
                                if char in string.punctuation or char.isnumeric():
                                    del_indices.append(j)
                filtered_sentence = [sentence[i] for i in range(0, len(sentence)) if i not in del_indices]
                filtered_tags = [tags[i] for i in range(0, len(sentence)) if i not in del_indices]
                filtered_pos = [pos[i] for i in range(0, len(sentence)) if i not in del_indices]
                filtered_dep = [dep[i] for i in range(0, len(sentence)) if i not in del_indices]
                f_abstracts.append(filtered_sentence)
                f_all_tags.append(filtered_tags)
                f_all_pos.append(filtered_pos)
                f_all_dep.append(filtered_dep)
            return f_abstracts, f_all_tags, f_all_pos, f_all_dep

        return abstracts, all_tags, all_pos_tags, all_dependencies

    def split_train_valid(self, data):
        valid_count = int(0.2*len(data))
        train = data[0:len(data)-valid_count]
        valid = data[len(data)-valid_count:]
        return train, valid

    def make_lookup(self):
        table = dict(zip(range(0, len(self.unique_tags) + 1), self.unique_tags))
        return table

    def get_vocabulary(self):
        train_vocab = self.get_unique_words(self.train_sentences)
        valid_vocab = self.get_unique_words(self.valid_sentences)
        test_vocab = self.get_unique_words(self.test_sentences)
        vocab = set(list(train_vocab) + list(test_vocab) + list(valid_vocab))
        table_arr = np.array(list(map(str.lower, vocab)))
        counter = Counter(table_arr)
        vocab_size = len(counter)
        vocabulary = [token for token, count in counter.most_common(vocab_size)]
        return vocabulary


class LSTMPrepper:
    def __init__(self):
        self.dl = Dataloader(train_path='train.json', test_path='test.json')
        self.tokenizer, self.embedding, self.x_train_padded, self.y_train_padded, self.x_test_padded, self.y_test_padded, self.max_len = self.tokenize()
        self.y_train = keras.utils.to_categorical(self.y_train_padded)
        self.y_test = keras.utils.to_categorical(self.y_test_padded)

    def find_max_sublist(self, data):
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
        GLOVE_DIM = 100 #181
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
            else:
                break
        return tokenizer, emb_matrix, x_train_padded, y_train_padded, x_test_padded, y_test_padded, pad_len


class BertPrepper(Dataset):
    def __init__(self, sentences, tags, pos_tags, dependencies, unique_tags, max_len, all_pos, all_deps):
        self.sentences = sentences
        self.tags = tags
        self.pos_tags = pos_tags
        self.dependencies = dependencies
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
        self.max_len = max_len
        self.len = len(sentences)
        self.label2id = {k: v for v, k in enumerate(unique_tags)}
        self.id2label = {v: k for v, k in enumerate(unique_tags)}
        nlp = spacy.load("en_core_web_sm")
        self.pos2cat = {k: v for v,k in enumerate(all_pos)}
        self.dep2cat = {k: v for v,k in enumerate(all_deps)}

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_labels = self.tags[idx]
        pos = self.pos_tags[idx]
        dep = self.dependencies[idx]
        tokenized_sentence, labels, pos_tags, dependencies = self.tokenize_with_tag_preservation(sentence=sentence, tags=word_labels, pos_tags=pos, dependencies=dep)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]  # add special tokens for bert
        labels.insert(0, "O")               # insert 0 for cls at the beginning of the sentence
        labels.insert(-1, "O")              # insert 0 for sep at the end of the sentence
        pos_tags.insert(0, "O")             # repeat this analogously for pos tags and dependencies
        pos_tags.insert(-1, "O")
        dependencies.insert(0, "O")
        dependencies.insert(-1, "O")
        maxlen = self.max_len   # pad everything to the longest sentence in the dataset
        # in case somehow there is still a sentence longer than max_len, truncate it
        if len(tokenized_sentence) > maxlen:
            tokenized_sentence = tokenized_sentence[:maxlen]
            labels = labels[:maxlen]
            pos_tags = pos_tags[:maxlen]
            dependencies = dependencies[:maxlen]
        else:   # pad to max_len
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(maxlen - len(tokenized_sentence))]
            labels = labels + ["O" for _ in range(maxlen - len(labels))]
            pos_tags = pos_tags + ["O" for _ in range(maxlen - len(pos_tags))]
            dependencies = dependencies + ["O" for _ in range(maxlen - len(dependencies))]
        attention_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]  # create attention mask, ignore paddings
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)      # tokens to ids
        label_ids = [self.label2id[label] for label in labels]
        pos_ids = [22 if pos == 'O' else self.pos2cat[pos] for pos in pos_tags]
        dep_ids = [46 if dep == 'O' else self.dep2cat[dep] for dep in dependencies]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attention_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long),
            'pos_embs': torch.tensor(pos_ids, dtype=torch.long),
            'dep_embs': torch.tensor(dep_ids, dtype=torch.long)
        }

    def tokenize_with_tag_preservation(self, sentence, tags, pos_tags, dependencies):
        """
        Given a sentence and its corresponding tags, tokenize it and preserve its original tags by adding
        the tag to each sub-word token.
        :param sentence: the sentence that is supposed to be tokenized
        :param tags: label tags for NER
        :param pos_tags: POS tags generated with spacy
        :param dependencies: Dependencies generated with spacy
        :return: tokenized version of the aforementioned parameters which fits the amount of sub-word tokens
        """
        tokenized_sentence = []
        tokenized_tags = []
        tokenized_pos_tags = []
        tokenized_dependencies = []
        for word, tag, pos_tag, dep in zip(sentence, tags, pos_tags, dependencies):
            tokenized = self.tokenizer.tokenize(word)
            word_token_count = len(tokenized)                           # count the sub-word tokens
            tokenized_sentence.extend(tokenized)
            tokenized_tags.extend(word_token_count * [tag])             # add label and pos tag for every token that
            tokenized_pos_tags.extend(word_token_count * [pos_tag])     # belonged to the originally labeled word
            tokenized_dependencies.extend(word_token_count*[dep])
        return tokenized_sentence, tokenized_tags, tokenized_pos_tags, tokenized_dependencies

    def __len__(self):
        return self.len


#lstm_prep = LSTMPrepper()