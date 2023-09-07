from transformers import BertTokenizer
from pretrained_transformer import Dataset
import preproccessing as prep
import pretrained_transformer as pt
import torch
import bert_crf as bc
# This is a sample Python script.
from torch.utils.data import Dataset, DataLoader
#import tensorflow as tf
from transformers import BertTokenizer, BertConfig, BertForTokenClassification

#https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb


PYTORCH_NO_CUDA_MEMORY_CACHING = 1
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #tokenizer = BertTokenizer.from_pretrained('bert_model-base-uncased')
    #dl = prep.Dataloader(train_path='train.json', test_path='test.json')
    #train_set = pt.BertPrepper(sentences=dl.train_sentences, tags=dl.train_tags, unique_tags=dl.unique_tags, tokenizer=tokenizer, max_len=128)
    #test_set = pt.BertPrepper(sentences=dl.test_sentences, tags=dl.test_tags, unique_tags=dl.unique_tags, tokenizer=tokenizer, max_len=128)
    #print(train_set[0])

    #pt.BertBase(filter_dataset=False)
    #pt.BertBase(filter_dataset=False)
    bc.BertExtended(name='bert_crf', filter_dataset=False)
