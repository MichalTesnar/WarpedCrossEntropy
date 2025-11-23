import torch.nn as nn
from constants import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE) # flatten [batch_size, 1, 28, 28] to [batch_size, 28x28]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out