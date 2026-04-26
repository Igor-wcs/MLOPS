"""
Construção flexível de modelos LSTM usando o padrão Factory Method.
"""
import torch.nn as nn
from src.models.lstm_params import LSTMParams

class LSTM(nn.Module):
    """Modelo LSTM flexível. Não instanciar diretamente — usar LSTMFactory."""
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
                if i + 1 < len(self.layers) and not isinstance(self.layers[i + 1], nn.LSTM):
                    x = x[:, -1, :] # Pega o último passo temporal
            else:
                x = layer(x)
        return x

class LSTMFactory:
    """Factory para criar modelos LSTM com arquiteturas customizáveis."""
    def __init__(self, layer_config: dict[str, str], params: LSTMParams) -> None:
        self.layer_config = layer_config
        self.params = params

    def get_layer(self, layer_name: str, input_size: int | None = None) -> nn.Module:
        if input_size is None:
            input_size = self.params.input_size

        match layer_name:
            case "LSTM":
                return nn.LSTM(
                    input_size, self.params.hidden_size, self.params.num_layers,
                    batch_first=self.params.batch_first,
                    dropout=self.params.dropout if self.params.num_layers > 1 else 0.0,
                )
            case "Linear":
                return nn.Linear(input_size, self.params.output_size)
            case "Dropout":
                return nn.Dropout(p=self.params.dropout)
            case _:
                raise ValueError(f"Camada '{layer_name}' não suportada.")

    def create(self) -> LSTM:
        """Constrói o modelo LSTM conforme a configuração do YAML."""
        layers: list[nn.Module] = []
        current_input_size = self.params.input_size

        for layer_type in self.layer_config.values():
            if layer_type == "LSTM":
                layer = self.get_layer("LSTM", input_size=current_input_size)
                current_input_size = self.params.hidden_size
            elif layer_type == "Linear":
                layer = self.get_layer("Linear", input_size=current_input_size)
                current_input_size = self.params.output_size
            else:
                layer = self.get_layer(layer_type)
            layers.append(layer)

        return LSTM(layers)