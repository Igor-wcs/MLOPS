import torch
import torch.nn as nn
from src.models.lstm_params import LSTMParams

class StockLSTM(nn.Module):
    """Arquitetura LSTM para predição de séries temporais."""
    def __init__(self, params: LSTMParams):
        super(StockLSTM, self).__init__()
        self.hidden_size = params.hidden_size
        self.num_layers = params.num_layers
        
        self.lstm = nn.LSTM(
            params.input_size, 
            params.hidden_size, 
            params.num_layers, 
            batch_first=True, 
            dropout=params.dropout if params.num_layers > 1 else 0
        )
        self.fc = nn.Linear(params.hidden_size, params.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def get_model(params: LSTMParams) -> nn.Module:
    """Factory para instanciar o modelo StockLSTM."""
    return StockLSTM(params)