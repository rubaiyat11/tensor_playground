import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(1, 1)


    def forward(self, x):
        return self.linear(x)
    

x = torch.arange(0,20).float().unsqueeze(1)
y = 3 * x + 5

train_x = x[:17]
train_y = y[:17]

test_x = x[17:]
test_y = y[17:]

model = LinearRegressionModel()

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
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

        print("Epoch:", epoch)
        print("Loss:", loss.item())
        print(model.state_dict())
        print()



torch.save(model.state_dict(), "linear_model.pth")

loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load("linear_model.pth"))

with torch.no_grad():
    test_predictions = loaded_model(test_x)

print("Test predictions:")
print(test_predictions)

print("Actual values:")
print(test_y)