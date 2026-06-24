import torch
from torch import nn


class MultiClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 16)

        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(16, 3)


    def forward(self, x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        return x
    

model = MultiClassifierModel()

x = torch.tensor([
    [1.,1.],
    [1.,2.],
    [2.,1.],

    [5.,5.],
    [5.,6.],
    [6.,5.],

    [9.,9.],
    [9.,10.],
    [10.,9.]
])

y = torch.tensor([
    0,
    0,
    0,

    1,
    1,
    1,

    2,
    2,
    2
])


train_length = int(len(x) * 0.8)

train_x = x[:train_length]
train_y = y[:train_length]

test_x = x[train_length:]
test_y = y[train_length:]

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


epochs = 500

for epoch in range(epochs):

    predictions = model(train_x)

    loss = loss_fn(
        predictions,
        train_y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(epoch)
        print("Loss: ", loss.item())



torch.save(model.state_dict(), "multiclassifiermodel.pth")

model.eval()


loaded_model = MultiClassifierModel()
loaded_model.load_state_dict(torch.load("multiclassifiermodel.pth"))

loaded_model.eval()

with torch.no_grad():
    test_predictions = loaded_model(test_x)

probabilities = torch.softmax(test_predictions, dim=1)

predicted_classes = probabilities.argmax(dim=1)

print(probabilities)

print(predicted_classes)

print(test_y)