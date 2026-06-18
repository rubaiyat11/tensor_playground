import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))


    
    def forward(self, x):
        return self.weight * x + self.bias
    


x = torch.arange(0, 11).float()

y = 2*x + 1

print(x)
print(y)

train_x = x[:8]
train_y = y[:8]

test_x = x[8:]
test_y = y[8:]

print(train_x)
print(train_y)

print(test_x)
print(test_y)

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

    if epoch % 10 == 0:
        
        print(f"Epch {epoch}")

        print(f"Loss: {loss}")

        print(model.state_dict())

        print()

print()

print("Final parameters")

print(model.state_dict())

predictions = model(test_x)

print()

print("Test predictions")

print(predictions)

print()

print("Actual values")

print(test_y)
