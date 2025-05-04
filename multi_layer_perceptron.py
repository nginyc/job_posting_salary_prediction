from sklearn.base import BaseEstimator
import torch
from torch import nn
from skorch import NeuralNetRegressor

class TensorTransformer(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return torch.tensor(X, dtype=torch.float32)

class Model(nn.Module):
    def __init__(self, num_hidden_layers: int, n_units_last: int, dropout_rate: float):
        super().__init__()

        if num_hidden_layers < 1 or num_hidden_layers > 4:
            raise ValueError("num_hidden_layers must be between 1 and 4")

        self.layers = nn.ModuleList()
        layer_sizes = [n_units_last * (2 ** i) for i in reversed(range(num_hidden_layers))]
        
        # Add hidden layers based on num_hidden_layers parameter
        for i in range(num_hidden_layers):
            layer_size = layer_sizes[i]
            self.layers.append(nn.LazyLinear(layer_size))
            self.layers.append(nn.BatchNorm1d(layer_size))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout_rate))

         # Output layer
        self.layers.append(nn.LazyLinear(1))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
            
        return X

class CustomNeuralNetRegressor(NeuralNetRegressor):
    def __init__(self, *args, lambda1=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda1 = lambda1

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        # L1 regularization for only the first layer
        loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.layers[0].parameters()])
        return loss
        