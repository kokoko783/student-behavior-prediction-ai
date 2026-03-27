import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, nhead=2, num_layers=1, dim_feedforward=64):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_size, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(dim_feedforward, num_classes)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x