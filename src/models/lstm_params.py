from pydantic import BaseModel, Field


class LSTMParams(BaseModel):
    """Hiperparâmetros para o modelo LSTM."""

    input_size: int = Field(default=1, description="Número de features de entrada")
    hidden_size: int = Field(default=64, ge=16, le=512)
    num_layers: int = Field(default=2, ge=1)
    output_size: int = Field(default=1)
    learning_rate: float = Field(default=0.001, gt=0)
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=32, ge=8)
    dropout: float = Field(default=0.2, ge=0.0, le=0.5)
