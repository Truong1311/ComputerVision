from unicodedata import numeric
import nltk
import torch
import contractions
import torch.nn as nn
from torch._utils.data import Dataset, DataLoader
from torch._utils.rnn import pad_sequence  
from collections import Counter
import pandas as pd
from PIL import Image
# define the start word, end word and unknown word and pad token
SOS = 0
EOS = 1
UNK = 2
PAD = 3

class Vocabulary:
    def __init__(self, freq_threshold):
        self.idx2w = {0: '<SOS>', 1: '<EOS>', 2: '<UNK>', 3: '<PAD>'}
        self.w2idx = {'<SOS>': 0, '<EOS>': 1, '<UNK>': 2, '<PAD>': 3}
        self.freq_threshold = freq_threshold
        self.idx = 4
    def __len__(self):
        return len(self.idx2w)
    def build_vocabulary(self, sentences):
        counter = Counter()
        
        for text in sentences:
            text = contractions.fix(text).lower()
            tokens = nltk.tokenize.word_tokenize(text)
            counter.update(tokens)
        vocab = [word for word, cnt in counter if cnt > self.freq_threshold]
        ## update w2idx and idx2w
        for word in vocab:
            if word not in self.w2idx:
                self.w2idx[word] = self.idx
                self.idx2w[self.idx] = word
                self.idx +=1
    def text2numeric(self, text):
        token = '<SOS>' + nltk.tokenize.word_tokenize(text) + '<EOS'
        return torch.from_numpy([self.w2idx[word] if word in self.w2idx else self.w2idx['UNK']
        for word in token
        ])
class Flickr8kDataset(Dataset):
    def __init__(self, path, transform = None, freq_threshold = 5):
        self.path = path
        self.imgpath = path + 'Images/'
        self.captionpath = path + 'captions.txt'
        self.transform = transform
        self.freq_threshold = freq_threshold

        self.df = pd.read_csv(self.captionpath)
        self.captions = self.df['caption']
        self.imgs = self.df['image']
        self.vocab = Vocabulary(self.freq_threshold)
        self.vocab.build_vocabulary(self.captions.values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        ## convert image to RGB color set( for resnet pretrained model)
        img = Image.open(self.img_path + self.imgs[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        numeric_token = self.vocab.text2numeric(caption)

        return img, numeric_token


