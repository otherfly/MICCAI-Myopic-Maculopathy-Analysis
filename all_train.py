from auto_encoder import Autoencoder
import torch
from timm import create_model
from models.resnet_simclr import ResNetSimCLR
from dataloader_all import train_loader, valid_loader
# from torchvision.models import vit_b_16
from vit_pytorch import SimpleViT
import torch.nn as nn
import datetime
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np


def classification_metrics(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    f1 = f1_score(y_true, y_pred, average='macro')
    spe = np.mean(specificity(y_true, y_pred))
    return dict(qwk=qwk, f1=f1, spe=spe)


def specificity(y_true: np.array, y_pred: np.array, classes: set = None):
    if classes is None:
        classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))
    specs = []
    for cls in classes:
        y_true_cls = np.array((y_true == cls), np.int32)
        y_pred_cls = np.array((y_pred == cls), np.int32)
        specs.append(recall_score(y_true_cls, y_pred_cls, pos_label=0))
    return specs


# AE_model=Autoencoder()
# state_dict=torch.load("/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/stack_autoencoder_10epoch_519.pth")
# AE_model.load_state_dict(state_dict)
# for name, params in AE_model.named_parameters():
#     params.requires_grad=False

Swin_trans_model = create_model('swin_base_patch4_window7_224', num_classes=5, pretrained=False, checkpoint_path="/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/swin_transformer_20240421-073851.pth") 
for name, params in Swin_trans_model.named_parameters():
    params.requires_grad=False

SimCLR_model = ResNetSimCLR(base_model='resnet50', out_dim=4096)
check_points = torch.load("/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/runs/Apr20_09-27-01_server-b/checkpoint_1000.pth.tar")
SimCLR_model.load_state_dict(check_points['state_dict'])
for name, params in SimCLR_model.named_parameters():
    params.requires_grad=False


train_model = SimpleViT(
    image_size = 32,
    patch_size = 2,
    num_classes = 5,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels= 53
)

state_dict=torch.load("/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/final_models/swin_sim/final_model_20240424-233243_999.pth")
train_model.load_state_dict(state_dict)
# print(train_model)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

train_model.to(device)
# AE_model.to(device)
Swin_trans_model.to(device)
SimCLR_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(train_model.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

epochs = 400
writer = SummaryWriter("/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/final_models/swin_sim/time_2")

for epoch in range(epochs):
    running_loss = 0.0
    running_corrects = 0
    for image1,image2,image3,labels in train_loader:
        image1=image1.to(device)
        # image2=image2.to(device)
        image3=image3.to(device)
        # feature_1 = AE_model(image2)
        feature_2 = Swin_trans_model.forward_features(image1)
        # feature_2 = Swin_trans_model(image1)
        feature_3 = SimCLR_model(image3)
        # feature_1 = torch.reshape(feature_1,(32,-1,32,32))
        feature_2 = torch.reshape(feature_2,(32,-1,32,32))
        feature_3 = torch.reshape(feature_3,(32,-1,32,32))

        feature_all = torch.cat((feature_2,feature_3),dim=1).to(device)
        labels = labels.to(device)

        outputs = train_model(feature_all)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * feature_all.size(0)
        running_corrects += torch.sum(preds == labels.data)

    if epoch>=10:
        scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    pred = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    pred = np.argmax(pred,axis=1)
    metrics = classification_metrics(labels,pred)

    steps = epoch*len(train_loader.dataset)/32
    writer.add_scalar('loss', epoch_loss, global_step=steps)
    writer.add_scalar('qwk', metrics["qwk"], global_step=steps)
    writer.add_scalar('f1', metrics["f1"], global_step=steps)
    writer.add_scalar('spe', metrics["spe"], global_step=steps)


    if epoch%100 == 99:
        model_dir = "/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/final_models/swin_sim/time_2"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_filename = os.path.join(model_dir, f'final_model_{start_time}_{epoch}.pth')
        torch.save(train_model.state_dict(), model_filename)
        





