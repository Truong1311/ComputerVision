import torch
import torch.nn as nn
import torchvision.models as models

class EncodeCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncodeCNN, self).__init__()
        self.embed_size = embed_size
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        moduls = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*moduls)
        self.emb = nn.Linear(resnet.fc.in_features, self.embed_size)
        init_weights()
    
    def init_weights(self):
        # khởi tạo trọng số và bias sử dụng phân phối chuẩn để mô hình dễ hội tụ.
        self.emb.weight.data.normal_(0, 0.02)
        self.emb.bias.data.fill_(0)
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.emb(features)
        return features
class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers = 1):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        ## embeding layer
        self.embeding = nn.Embedding(self.vocab_size, self.embed_size)
        ## lstm layer
        self.lstm = nn.LSTM(
            self.embed_size,
            self.hidden_size,
            self.num_layers,
            batch_first= True
        )
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        init_weights()
    def init_weights(self):
        self.embeding.weight.data.normal_(-0.01, 0.01)
        self.embedding.bias.data.fill_(0)
    def forward(self, images, features):
        pass
        


