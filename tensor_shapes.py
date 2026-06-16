import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.arange(1, 13)

print(x)

print(x.shape)
print(x.size())


print(x.reshape(3, 4))


x = torch.tensor([[5]])

print(x.shape)

print(x.squeeze())


shape = [1,2]

matrix = torch.tensor([
    [2, 3]
])

matrix_shape = matrix.shape

print(matrix_shape)
print(shape)


state = torch.tensor([2.0, -1.0])

print(state.shape)

state_unsqueezed = state.unsqueeze(dim=0)

print(state_unsqueezed)
print(state_unsqueezed.shape)



