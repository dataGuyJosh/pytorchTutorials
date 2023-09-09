import torch
import numpy as np

'''
Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model's parameters.

Tensors are similar to NumPy's ndarrays, except that tensors can run on GPUs or other hardware accelerators.
In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data (see Bridge with NumPy).
Tensors are also optimized for automatic differentiation (we'll see more about that later in the Autograd section).
'''

'''Initializing a Tensor'''
# from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
# from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# from another tensor
# retains the properties of x_data
x_ones = torch.ones_like(x_data)
# overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float)
# with random or constant values
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f'''
    Ones Tensor:
        {x_ones}
    Random Tensor:
        {x_rand}
    Random Tensor:
        {rand_tensor}
    Ones Tensor:
        {ones_tensor}
    Zeros Tensor:
        {zeros_tensor}
''')
# attributes
tensor = torch.rand(3, 4)
print(f'''
    Shape of tensor:            {tensor.shape}
    Datatype of tensor:         {tensor.dtype}
    Device tensor is stored on: {tensor.device}
''')
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
# operations
tensor = torch.ones(4, 4)
print(f'''
    First row:      {tensor[0]}
    First column:   {tensor[:, 0]}
    Last column:    {tensor[..., -1]}
''')
# update second column
tensor[:, 1] = 0
print('Updating second column:\n',tensor)
# joining (concatenating)
print('Joining tensors:\n',torch.cat([tensor, tensor], dim=1))


'''Arithmetic Operations'''
# matrix multiplication between two tensors
# y1, y2, y3 will have the same value
# "tensor.T" returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# single-element tensors can be converted to python numerical values using "item()"
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# in-place operations are denoted by a "_" suffix
print(tensor)
tensor.add_(5)
print(tensor)


'''
Bridge with NumPy: Tensors on the CPU and NumPy arrays can share their
underlying memory locations, and changing one will change the other.
'''
# tensor to numpy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
# changes in tensor are reflected in numpy array & vice versa
t.add_(1)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
# numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)
