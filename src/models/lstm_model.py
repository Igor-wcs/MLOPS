import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Arquitetura LSTM para predição de séries temporais financeiras.
    Suporta inicialização de estados e aceleração via hardware.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # O PyTorch exige que num_layers > 1 para aplicar dropout na LSTM
        valid_dropout = dropout_rate if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=valid_dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passo forward da rede com inicialização explícita de estados."""
        # Inicializa hidden e cell states com zeros no mesmo device do input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Extrai apenas o último passo temporal da sequência (Last-Step)
        ultimo_estado = self.fc(out[:, -1, :])
        return ultimo_estado
