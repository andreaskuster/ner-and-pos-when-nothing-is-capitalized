from torch import nn


class TrueCaser(nn.Module):
    def __init__(self, num_tokens: int, size_features: int):
        super().__init__()
        self.encoder = nn.Embedding(num_tokens, size_features)
        self.lstm = nn.LSTM(
            input_size=size_features,
            hidden_size=size_features,
            num_layers=2,
            bidirectional=True
        )
        self.decoder = nn.Linear(2 * size_features, 2)

    def forward(self, x, hidden=None):

        x = self.encoder(x)
        x = x.view([x.shape[1], x.shape[0], x.shape[2]])
        x, hidden = self.lstm(x, hidden)
        x = self.decoder(x)
        return x, hidden
