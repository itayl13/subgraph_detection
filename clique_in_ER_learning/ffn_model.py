
import torch
from torch.nn.modules.module import Module
from torch.nn import ModuleList
from torch.nn import functional
from torch.nn import Linear


class FFNClique(Module):
    def __init__(self, input_shape, hidden_layers, dropouts):
        super(FFNClique, self).__init__()
        hidden_layers = [input_shape] + hidden_layers + [1]    # input_dim, hidden_layer0, ..., hidden_layerN, 1
        self._layers = ModuleList([Linear(first, second)
                                   for first, second in zip(hidden_layers[:-1], hidden_layers[1:])])
        self._activations = [torch.nn.functional.relu] * (len(hidden_layers) - 1)  # Activation functions from input layer to last hidden layer
        self._dropouts = dropouts

    def forward(self, x):
        layers = list(self._layers)
        layer_for_w = []
        for i, layer in enumerate(layers[:-1]):
            layer_out = self._activations[i](layer(x))
            layer_for_w.append(layer_out)
            x = functional.dropout(layer_out, self._dropouts[i], training=self.training)
        x = layers[-1](x)
        return torch.sigmoid(x)

    @property
    def linear_layer_params(self):
        return list(self._layers)
