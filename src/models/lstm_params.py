"""
Parâmetros validados para construção de modelos LSTM.
Usa Pydantic garantir que ninguém passa hiperparâmetros inválidos (ex: dropout negativo).
"""
from pydantic import BaseModel, Field

class LSTMParams(BaseModel):
    """Hiperparâmetros para o modelo LSTM."""
    input_size: int = Field(ge=1, description="Número de features de entrada (Ex: 11 para PETR4)")
    hidden_size: int = Field(default=64, ge=1)
    num_layers: int = Field(default=2, ge=1)
    output_size: int = Field(default=1, ge=1, description="Dimensão de saída (1 para regressão de preço)")
    batch_first: bool = Field(default=True)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)