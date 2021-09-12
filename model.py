import torch
import torchvision
import torch.nn as nn

# %%
num_classes = 8
model = torchvision.models.resnet50(pretrained=True)
fc_inputs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(fc_inputs, 256),
                         nn.Linear(256, 8))
model = model.cuda()
print(model)

# %%
input = torch.rand((3, 3, 224, 224))
input = input.cuda()
print(input.shape)
output = model(input)
print(output.shape)
print(output.argmax(dim=1))
