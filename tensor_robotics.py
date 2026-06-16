import torch

state = torch.tensor([
    2.5,
    -1.2,
    0.3,
    0.8
])

print(state)
print(state.shape)


batch_state = state.unsqueeze(0)

print(batch_state)
print(batch_state.shape)


column_state = state.unsqueeze(1)

print(column_state)
print(column_state.shape)


print(state[0])
print(state[1])
print(state[2])
print(state[3])

state_min = torch.min(state)
state_max = torch.max(state)
state_mean = torch.mean(state)
state_sum = torch.sum(state)


state_reshaped = state.reshape(2,2)

print(state_reshaped)



A = torch.tensor([
    [1.0, 0.1],
    [0.0, 1.0]
])

x = torch.tensor([
    2.0,
    -1.0
])

next_state = A @ x

print(next_state)
print(next_state.shape)