import torch
import torchvision.models as models

'''
Saving and Loading model weights
PyTorch models store the learned parameters in an internal state dictionary "state_dict".
These can be persisted via the torch.save method.
'''
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'data/model_weights.pth')

'''
Load model weights by creating an instance of the model
then loading parameters using "load_state_dict()".
'''

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('data/model_weights.pth'))
model.eval()

'''
Saving and Loading Models with Shapes
When loading model weights, we needed to instantiate the model first
as the class defines network structure. We can save the class structure together
with the model by passing "model" (and not "model.state_dict()) to the saving function.
'''
torch.save(model,'data/model.pth')
# load saved model
model = torch.load('data/model.pth')