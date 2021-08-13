import torch.nn as nn

class ResidualBlockWithBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, input_dim)
        self.batch_norm_1 = nn.BatchNorm1d(input_dim)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.batch_norm_1(x)
        out = self.activation(out)
        out = self.linear_1(out)
        out = self.batch_norm_2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = out + x
        return out

class ResidualBlockWithLayerNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.layer_norm_1(x)
        out = self.activation(out)
        out = self.linear_1(out)
        out = self.layer_norm_2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = out + x
        return out