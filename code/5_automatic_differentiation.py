'''
Automatic differentiation with torch.autograd
When training neural networks, the most frequently used algorithm is back propagation.
In this algorithm, parameters (model weights) are adjusted according to the gradient
of the loss function with respect to the given parameter.

To compute those gradients, PyTorch has a built-in differentiation engine called
"torch.autograd". It supports automatic computation of gradient for any
computational graph.

Consider the simplest one-layer neural network, with input x, parameters w and b,
and some loss function. It can be defined in PyTorch as follows:
'''

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

'''
This code defined the following computational graph:
x > * > + > z > CD > loss
    ^w  ^b
In this network, w and b are parameters which need to optimize.
Thus we need to be able to compute the gradient of loss function
with respect to those variables. In order to do that we set the
"requires_grad" property of those tensors.

A function that we apply to tensors to construct computational graph
is in fact an object of class "Function". This object knows how to compute
the function in the forward direction, and also how to compute its derivative
during the backward propagation step. A reference to the backward propagation function
is stored in the "grad_fn" property of a tensor.
'''

print(f'''
Gradient function for z: {z.grad_fn}
Gradient function for loss: {loss.grad_fn}
''')

'''
Computing Gradients
To optimize weights of parameters in the neural network, we need to
compute the derivatives of our loss function with respect to parameters,
specifically dloss/dw and dloss/db under some fixed values of x and y.
To compute those derivatives, we call "loss.backward()" and retrieve
the values from w.grad and b.grad.
'''

loss.backward()
print(w.grad, b.grad, sep='\n')


'''
Disabling Gradient Tracking
By defaul, all tensors with "requires_grad=True" are tracking their
computational history and support gradient computation. However, there are cases
where we have trained the model and just want to apply it to some input data
i.e. we only want to do forward computations through the network.
We can stop tracking computations by surrounding our computation code with a "torch.no_grad()" block:
'''

z = torch.matmul(x, w)+b
print(z.requires_grad)
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# another way to achieve the same result is to use "detach()"
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

'''
Reasons to disable gradient tracking:
- mark some parameters in a neural network as "frozen"
- speed up computation when only doing forward pass,
  computations on tensors are more efficient when not tracking gradients

More on computational graphs
Conceptually, autograd keeps a record of data (tensors) and all executed operations
(along with the resulting new tensors) in a directed acyclic graph (DAG)
consisting of Function objects. In this DAG, leaves are the input tensors,
roots are the output tensors. By tracing this graph from roots to leaves,
you can automatically compute gradients using the chain rule.

The backward pass starts when ".backward()" 
is called on the DAG root, at which put autograd:
- computes the gradients from each .grad_fn
- accumulates them in the respective tensor's ".grad" attribute
- using the chain rule, propagates to leaf tensors
'''