import torch


x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([3.0, 5.0, 7.0, 9.0])

weight = 1.5
bias = 1.0

prediction = weight * x + bias

print(prediction)

error = prediction - y

print(error)

loss = torch.abs(error)

print(loss)

mean_loss = torch.mean(loss)

print(mean_loss)

