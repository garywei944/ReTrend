from torch import nn


class SemNetModel(nn.Module):
    def __init__(
            self,
            in_dim,
            h1_dim,
            h2_dim
    ):
        super(SemNetModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, h1_dim),  # 17 properties
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, 1)
        )

    def forward(self, x):
        return self.model(x)
