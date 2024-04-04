from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers_source,
            num_layers_target,
            n_nodes_source,
            n_nodes_target,
            output_size,
            dropout,
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
        self.hidden_size_source = n_nodes_source
        self.hidden_size_target = n_nodes_target
        self.num_layers_source = num_layers_source
        self.num_layers_target = num_layers_target
        self.output_size = output_size
        self.dropout = dropout
        self.day_index = day_index
        if num_layers_source != 0:
            self.source_lstm = nn.LSTM(input_size, n_nodes_source, num_layers_source, dropout=dropout, batch_first=True)
            self.target_lstm = nn.LSTM(n_nodes_source, n_nodes_target, num_layers_target, batch_first=True)
        else:
            self.target_lstm = nn.LSTM(input_size,n_nodes_target, num_layers_target, batch_first=True)
        self.linear = nn.Linear(in_features=n_nodes_target, out_features=output_size)

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
        if self.num_layers_source != 0:
            hidden, _ = self.source_lstm(input_new, None)
            hidden, _ = self.target_lstm(hidden, None)
        else:
            hidden, _ = self.target_lstm(input_new, None)
        if hidden.dim() == 2:
            hidden = hidden[-1, :]
        else:
            hidden = hidden[:, -1, :]
        output = self.linear(hidden)
        if self.day_index is not None:
            output[night_mask] = 0
            #output[output<0] = 0 # Other physical post-processing, we know power cannot be below zero
        return output
