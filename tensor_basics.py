import torch

scalar = torch.tensor(7)

print(scalar)
print(scalar.ndim)
print(scalar.shape)


vector = torch.tensor([1, 2, 3])

print(vector)
print(vector.ndim)
print(vector.shape)

matrix = torch.tensor([
    [1, 2],
    [3, 4]
])


print(matrix)
print(matrix.ndim)
print(matrix.shape)



A = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

B = torch.tensor([
    [5.0],
    [6.0]
])

C = torch.matmul(A, B)

print(C)


device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


A = A.to(device)
B = B.to(device)

C = torch.matmul(A, B)

print(C)
print(C.device)


D = torch.tensor([
    [1, 2],
    [3, 4],
    [5, 6]
])

print(torch.matmul(D, A))