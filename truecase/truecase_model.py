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

        # Apply LSTM
        # * Transposes are necessary due to the fact that LSTMs
        # * takes slightly different shape of data in (and out).
        x = x.transpose(0, 1)
        x, hidden = self.lstm(x, hidden)
        x = x.transpose(0, 1)

        x = self.decoder(x)
        return x, hidden
