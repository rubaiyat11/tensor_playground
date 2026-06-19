import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    

    
model = LinearRegressionModel()

x = torch.arange(0, 25).float().unsqueeze(1)
y = 4 * x - 2

train_x = x[:21]
train_y = y[:21]

test_x = x[21:]
test_y = y[21:]


loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
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



torch.save(model.state_dict(), "linear_model_2.pth")

loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load("linear_model_2.pth"))

loaded_model.eval()

with torch.no_grad():
    test_predictions = loaded_model(test_x)

print("Test Predictions:")
print(test_predictions)

print("Actual Values:")
print(test_y)