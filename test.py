import torch
import torchvision
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_generator import get_labels
from dataset import *

# %%


model = torchvision.models.resnet50(pretrained=True)
fc_inputs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(fc_inputs, 256),
                         nn.Linear(256, 8))
model = model.cuda()
print(model)

# %%
# Load state dictionary
model.load_state_dict(torch.load("./models/resnet_dict_40.pt"))

# %%
label_path = "./labels.json"
label_dirt = get_labels(label_path)
# %%
img = Image.open("./7_19.jpg")
img.show()
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

# %%
BATCH_SIZE = 30
valid_path = "./train.csv"
valid_dataset = GarbageData(valid_path, label_path, train_flag=False)
valid_dataset_size = len(valid_dataset)
valid_dataloader = DataLoader(valid_dataset,
                              BATCH_SIZE,
                              shuffle=False)

sum_accuracy = 0
for data in valid_dataloader:
    img, target, _ = data
    img = img.cuda()
    target = target.cuda()
    print(img.shape)

    output = model(img)
    accuracy = (output.argmax(dim=1) == target).sum()
    sum_accuracy += accuracy
    print(accuracy.item()/BATCH_SIZE)
print("acc:", sum_accuracy/valid_dataset_size)
