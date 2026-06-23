import torch
from torch import nn


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 32)
        
        self.layer2 = nn.Linear(32, 16)

        self.layer3 = nn.Linear(16, 8)

        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        x = self.relu(x)

        x = self.layer3(x)

        return x
    
model = BinaryClassifier()

x = torch.tensor([
    [1.,1.],
    [2.,1.],
    [2.,2.],
    [3.,2.],
    [4.,3.],
    [5.,4.],
    [6.,5.],
    [7.,6.]
])


y = torch.tensor([
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [1.],
    [1.]
])

train_length = int(len(x) * 0.8)

train_x = x[train_length:]
train_y = y[train_length:]

test_x = x[:train_length]
test_y = y[:train_length]

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

epochs = 500

for epoch in range(epochs):

    predictions = model(x)


