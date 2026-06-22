import torch
from torch import nn


class DeepRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()


    def forward(self, x):
        
        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        x = self.relu(x)

        x = self.layer3(x)

        return x
    

model = DeepRegressionModel()

x = torch.arange(0, 57).float().unsqueeze(1)

y = 7 * x - 3

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
        print(f"Epoch: {epoch}")
        print(f"Loss: {loss.item()}")
        print(model.state_dict())
        print()


torch.save(model.state_dict(), "deepregressionmodel.pth")

model.eval()

loaded_model = DeepRegressionModel()
loaded_model.load_state_dict(torch.load("deepregressionmodel.pth"))

loaded_model.eval()

with torch.no_grad():
    test_predictions = loaded_model(test_x)


print("Test Predictions:")
print(test_predictions)

print("Actual Values:")
print(test_y)