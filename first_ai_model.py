import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    
    def forward(self, x):
        return self.weight * x + self.bias
    

model = LinearRegressionModel()

print(model)
print(model.weight)
print(model.bias)

x = torch.tensor([1.0, 2.0, 3.0])

prediction = model(x)

print(prediction)
print(prediction.shape)

print(list(model.parameters()))
print(model.state_dict())