from torch import nn


class LSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers_source,
            num_layers_target,
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

        self.source_lstm = nn.LSTM(input_size, hidden_size, num_layers_source, dropout=dropout, batch_first=True)
        self.target_lstm = nn.LSTM(hidden_size, hidden_size, num_layers_target, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input):
        """
        Forward method for the LSTM layer. I.e. how input gets processed
        :param input: the input tensor
        :return: output tensor
        """
        hidden, _ = self.source_lstm(input, None)
        hidden, _ = self.target_lstm(hidden, None)
        if hidden.dim() == 2:
            hidden = hidden[-1, :]
        else:
            hidden = hidden[:, -1, :]
        output = self.linear(hidden)
        return output
