import torch.nn as nn


class ModeloLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(ModeloLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, dropout=dropout_rate
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        ultimo_estado = self.fc(
            lstm_out[:, -1, :]
        )  # Pega apenas o último passo temporal
        return ultimo_estado
