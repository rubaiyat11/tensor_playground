import torch


sample1 = torch.tensor([1.0, 2.0, 3.0])
sample2 = torch.tensor([4.0, 5.0, 6.0])
sample3 = torch.tensor([7.0, 8.0, 9.0])
sample4 = torch.tensor([10.0, 11.0, 12.0])

print(sample1.shape)
print(sample2.shape)
print(sample3.shape)
print(sample4.shape)


data = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
])

print(data)
print(data.shape)


x = data[2]

print(x)
print(x.shape)

x_batch = x.unsqueeze(0)

print(x_batch)
print(x_batch.shape)

x_col = x.unsqueeze(1)

print(x_col)
print(x_col.shape)

#Extracting 1st element from first sample from data
element_1 = data[0,0]
#Extracting 2nd element from last sample from data
element_2 = data[3, 2]

print(element_1)
print(element_2)

