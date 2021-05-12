"""
Class for classification model. 
"""

import torch


class LogisticRegressor(torch.nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_out_classes: int):
        super(LogisticRegressor, self).__init__()

        self.hidden = torch.nn.Linear(in_features, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_out_classes)

        self.activation = torch.nn.Softmax(dim=1)
    
    def forward(self, X: torch.tensor) -> torch.tensor:
        hidden        = self.hidden(X)
        out           = self.out(hidden)
        activated_out = self.activation(out)

        return activated_out
