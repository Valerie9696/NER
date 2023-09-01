from transformers import BertTokenizer
from pretrained_transformer import Dataset
import preproccessing as prep
import pretrained_transformer as pt
import torch
# This is a sample Python script.
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from transformers import BertTokenizer, BertConfig, BertForTokenClassification

#https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #dl = prep.Dataloader(train_path='train.json', test_path='test.json')
    #train_set = pt.BertPrepper(sentences=dl.train_sentences, tags=dl.train_tags, unique_tags=dl.unique_tags, tokenizer=tokenizer, max_len=128)
    #test_set = pt.BertPrepper(sentences=dl.test_sentences, tags=dl.test_tags, unique_tags=dl.unique_tags, tokenizer=tokenizer, max_len=128)
    #print(train_set[0])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # print('allocated memory: ', torch.cuda.memory_allocated(device=device))
        print(torch.cuda.current_device())
        print('START: free and total memory: ', torch.cuda.mem_get_info(device=torch.cuda.current_device()))
    pt.Model()


    #for i in range(0, len(dl.train_sentences)):
     #   pt.tokenize_and_preserve_labels(sentence=dl.train_sentences[i], text_labels=dl.train_tags[i])
    #padded = prep.pad(train, test)
    a=0

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
