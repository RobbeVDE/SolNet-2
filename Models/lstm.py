from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers_source,
            num_layers_target,
            day_index,
            output_size,
            dropout):
        """
        Simple LSTM model made in pytorch
        :param input_size: the size of the input (based on the lags provided)
        :param hidden_size: the hidden layer sizes
        :param num_layers: the number of layers in the LSTM (each of size hidden_size)
        :param output_size: the forecast window (f.e. 24 means 'forecast 24 hours')
        :param dropout: the dropout parameter used for training, to avoid overfitting
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers_source = num_layers_source
        self.num_layers_target = num_layers_target
        self.output_size = output_size
        self.dropout = dropout
        self.day_index = day_index

        self.source_lstm = nn.LSTM(input_size, hidden_size, num_layers_source, dropout=dropout, batch_first=True)
        self.target_lstm = nn.LSTM(hidden_size, hidden_size, num_layers_target, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input):
        """
        Forward method for the LSTM layer. I.e. how input gets processed
        :param input: the input tensor
        :return: output tensor
        """
        night_mask = input[:,:,self.day_index]
        night_mask = night_mask.bool()
        night_mask = ~night_mask #Now it was True when day
        if self.day_index == input.size(2): #If is_day is last feature we don't have to concat 2 strings
            bla = input[:,:,:self.day_index]
        else:
            one = input[:,:,:self.day_index]
            two = input[:,:,self.day_index+1:]
            bla = torch.cat((one, two), dim=2)
        hidden, _ = self.source_lstm(bla, None)
        hidden, _ = self.target_lstm(hidden, None)
        if hidden.dim() == 2:
            hidden = hidden[-1, :]
        else:
            hidden = hidden[:, -1, :]
        output = self.linear(hidden)
        output[night_mask] = 0
        return output
