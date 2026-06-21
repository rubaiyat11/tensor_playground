import torch
from torch import nn


class TinyNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 4)

        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(4, 1)

    
    def forward(self, x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        return x
    

model = TinyNN()

x = torch.tensor([
    [1., 2.],
    [2., 3.],
    [3., 4.],
    [4., 5.]
])

y = torch.tensor([
    [3.],
    [5.],
    [7.],
    [9.]
])


train_size = int(0.8 * len(x))

train_x = x[:train_size]
train_y = y[:train_size]

test_x = x[train_size:]
test_y = y[train_size:]


loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

epochs = 500

for epoch in range(epochs):

    model.train()

    predictions = model(train_x)

    loss = loss_fn(
        predictions,
        train_y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if epoch % 50 == 0:
        print("Epoch:", epoch)
        print("Loss:", loss.item())
        print(model.state_dict())
        print()

model.eval()

torch.save(model.state_dict(), "tiny_NN.pth")


loaded_model = TinyNN()
loaded_model.load_state_dict(torch.load("tiny_NN.pth"))


loaded_model.eval()

with torch.no_grad():
    test_predictions = loaded_model(test_x)

print("Test Predictions:")
print(test_predictions)

print("Actual Values:")
print(test_y)