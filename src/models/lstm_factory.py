import torch.nn as nn
from src.models.lstm_model import StockLSTM
from src.models.lstm_params import LSTMParams


def get_model(params: LSTMParams) -> nn.Module:
    """
    Factory para instanciar a arquitetura StockLSTM.
    Utiliza as restrições e validações do Pydantic (LSTMParams).
    """
    return StockLSTM(
        input_size=params.input_size,
        hidden_size=params.hidden_size,
        output_size=params.output_size,
        num_layers=params.num_layers,
        dropout_rate=params.dropout,
    )
