import torch.nn as nn

class LSTMFakeClassifier(nn.Module):
    '''
    A simple LSTM neural network.
    Allows for multiple layers and dropouts.
    When num_layers = 1, dropout has to be zero.
    '''

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, dropout):

        super(LSTMFakeClassifier, self).__init__()

        self.word_dict = None

        self.embed  = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0) #give paddin_idx = 0 to ignore 0 indices
        self.lstm   = nn.LSTM(embedding_dim, hidden_dim, num_layers = num_layers, 
                              dropout = dropout, batch_first = False)
        
        self.hidden = nn.Linear(in_features = hidden_dim, out_features = 1) #1 output feature since binary
        self.sig    = nn.Sigmoid()

    def forward(self, x):
        
        x = x.t()
        
        length = x[0,:]
        text   = x[1:,:]
        
        x1    = self.embed(text)
        x2, _ = self.lstm(x1)
        x3    = self.hidden(x2)
        x3_   = x3[length - 1, range(len(length))]
        out   = self.sig(x3_.squeeze())

        return out


