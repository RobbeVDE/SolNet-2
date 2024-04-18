from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size,
            dropout,
            bidirectional = False,
            day_index = None):
        """
        Simple LSTM model made in pytorch   STM(input_size, optimizer_name, lr_target, n_layers_source, n_layers_target, n_nodes_source, n_nodes_target, forecast_period, dropout)
        :param input_size: the size of the input (based on the lags provided)
        :param hidden_size: the hidden layer sizes
        :param num_layers: the number of layers in the LSTM (each of size hidden_size)
        :param output_size: the forecast window (f.e. 24 means 'forecast 24 hours')
        :param dropout: the dropout parameter used for training, to avoid overfitting
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.day_index = day_index
        self.bd = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(in_features=hidden_size*2, out_features=1)
        else:
            self.linear = nn.Linear(in_features=hidden_size, out_features=output_size) #Times 2 bcs output of lstm is times 2
        

    def forward(self, input):
        """
        Forward method for the LSTM layer. I.e. how input gets processed
        :param input: the input tensor
        :return: output tensor
        """
        if self.day_index is not None:
            night_mask = input[:,:,self.day_index]
            night_mask = night_mask.bool()
            night_mask = ~night_mask #Now it was True when day
            if ((self.day_index+1) == input.size(2)): #If is_day is last feature we don't have to concat 2 tensors
                input_new = input[:,:,:self.day_index]
            else:
                one = input[:,:,:self.day_index]
                two = input[:,:,self.day_index+1:]
                input_new = torch.cat((one, two), dim=2)
        else:
            input_new = input
        
        hidden, _ = self.lstm(input_new, None)

        if not self.bd:                   
            if hidden.dim() == 2:
                hidden = hidden[-1, :]
            else:
                hidden = hidden[:, -1, :]

        output = self.linear(hidden)
        if self.bd:
            output = torch.squeeze(output, dim=2)
        if self.day_index is not None:
            output[night_mask] = 0
            #output[output<0] = 0 # Other physical post-processing, we know power cannot be below zero
        return output
