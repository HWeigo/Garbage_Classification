import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import models

from dataset import *
from dataset_generator import get_labels

if __name__ == '__main__':
    BATCH_SIZE = 64
    EPOCHS = 60

    label_path = "./labels.json"
    label_dirt = get_labels(label_path)
    num_classes = len(label_dirt)
    print("class size: {}".format(num_classes))

    train_path = "./train.csv"
    valid_path = "./valid.csv"

    train_dataset = GarbageData(train_path, label_path, train_flag=True)
    valid_dataset = GarbageData(valid_path, label_path, train_flag=False)
    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)
    print("training size: {}, validation size: {}".format(train_dataset_size, valid_dataset_size))

    num_batch = int(train_dataset_size / BATCH_SIZE)
    train_dataloader = DataLoader(train_dataset,
                                  BATCH_SIZE,
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  BATCH_SIZE,
                                  shuffle=False)

    model = torchvision.models.resnet50(pretrained=True)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(fc_inputs, 256),
                             nn.Linear(256, num_classes))
    model = model.cuda()
    print(model)

    lr_init = 0.0001
    lr_stepsize = 20
    weight_decay = 0.001
    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)

    writer = SummaryWriter('logs/resnet50')

    for epoch in range(EPOCHS):
        scheduler.step()

        model.train()
        for step, data in enumerate(train_dataloader):
            img, target, _ = data
            img = img.cuda()
            target = target.cuda()

            output = model(img)
            loss = loss_func(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print("[TRAIN] epoch: [{}: {}/{}], loss: {}".format(epoch,
                                                                    step,
                                                                    num_batch,
                                                                    loss.item()))

        model.eval()
        with torch.no_grad():
            total_loss = 0
            accuracy = 0
            for data in valid_dataloader:
                img, target, _ = data
                img = img.cuda()
                target = target.cuda()

                output = model(img)
                loss = loss_func(output, target)
                total_loss = total_loss + loss

                accuracy = accuracy + (output.argmax(1) == target).sum()

            writer.add_scalar("eval/accuracy", accuracy / valid_dataset_size, epoch)
            writer.add_scalar("eval/loss", total_loss, epoch)
            print("[TEST] epoch:{}, loss: {}, accuracy: {}".format(epoch,
                                                                   total_loss.item(),
                                                                   accuracy / valid_dataset_size))
            print("-------------------------------------------------------")

            # Save model every 10 epoch
            if epoch % 10 == 0:
                # torch.save(model.state_dict(), "models/resnet_dict_{}.pt".format(epoch))
                # Checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    'accuracy': accuracy
                }, "models/resnet_dict_{}.pt".format(epoch))
                print("[CHECKPOINT] epoch:{}".format(epoch))

    writer.close()
