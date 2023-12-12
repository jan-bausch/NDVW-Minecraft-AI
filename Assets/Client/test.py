import torch

# Creating a 2D tensor with dimensions 3x4
x = torch.tensor([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
])

print(x.shape)

# Reshaping the 2D tensor into a 4D tensor
new_shape = (3, 3, 2, 2)  # The total number of elements remains the same (3x4=12)


x_4d = x.view(new_shape)
print(x_4d)
print(x_4d.shape)

print(x_4d[0, 0, :, :])
print(x_4d[0, 1, :, :])
print(x_4d[0, 2, :, :])

print(x_4d[1, 0, :, :])
print(x_4d[1, 1, :, :])
print(x_4d[1, 2, :, :])

print(x_4d[2, 0, :, :])
print(x_4d[2, 1, :, :])
print(x_4d[2, 2, :, :])