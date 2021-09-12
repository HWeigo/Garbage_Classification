import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms
from dataset_generator import get_labels

#%%


model = torchvision.models.resnet50(pretrained=True)
fc_inputs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(fc_inputs, 256),
                         nn.Linear(256, 8))
model = model.cuda()
print(model)

#%%
# Load state dictionary
model.load_state_dict(torch.load("./models/resnet_dict_30.pt", ))

#%%
label_path = "./labels.json"
label_dirt = get_labels(label_path)
#%%
img = Image.open("./test.jpg")
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
img_tensor = transform(img)
# print(img_tensor.shape)
img_tensor = torch.reshape(img_tensor, (1, 3, 224, 224))
img_tensor = img_tensor.cuda()
output = model(img_tensor)
print(output)
label_idx = output.argmax(dim=1).item()
print("Predict:", label_dirt[label_idx])