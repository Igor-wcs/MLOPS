import torch
import torch.nn as nn

class ModeloLSTM(nn.Module):
    """
    Arquitetura LSTM para predição de séries temporais financeiras.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        num_layers: int, 
        dropout_rate: float
    ):
        super(ModeloLSTM, self).__init__()
        
        # O PyTorch exige que num_layers > 1 para aplicar dropout na LSTM
        valid_dropout = dropout_rate if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=valid_dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passo forward da rede."""
        lstm_out, _ = self.lstm(x)
        # Extrai apenas o último passo temporal da sequência
        ultimo_estado = self.fc(lstm_out[:, -1, :])
        return ultimo_estado