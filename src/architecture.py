import torch
import torch.nn as nn


# %% ######################################################
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        
    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(src)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1, dropout=0):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(output_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # activation function
        self.relu = nn.ELU()
        
    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output)
        prediction = self.relu(prediction)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, output_length):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.output_length = output_length
        
    def forward(self, src, trg=None):
        batch_size = src.shape[0]
        output_dim = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, self.output_length, output_dim).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        input = torch.zeros(batch_size, 1, output_dim).to(self.device)
        
        for t in range(self.output_length):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            input = output
        
        return outputs
