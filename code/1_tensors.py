import torch
import numpy as np

'''
Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model's parameters.

Tensors are similar to NumPy's ndarrays, except that tensors can run on GPUs or other hardware accelerators.
In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data (see Bridge with NumPy).
Tensors are also optimized for automatic differentiation (we'll see more about that later in the Autograd section).
'''

# Initializing a Tensor
# from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
# from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# from another tensor
x_ones = torch.ones_like(x_data)                    # retains the properties of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# with random or constant values
shape = (2,3,)
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
tensor = torch.rand(3,4)
print(f'''
    Shape of tensor:            {tensor.shape}
    Datatype of tensor:         {tensor.dtype}
    Device tensor is stored on: {tensor.device}
''')