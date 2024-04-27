from auto_encoder import Autoencoder
import torch
from timm import create_model
from models.resnet_simclr import ResNetSimCLR
from dataloader_all import train_loader, valid_loader,valid_loader_len
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
from torchvision import transforms


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


AE_model=Autoencoder()
state_dict=torch.load("/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/stack_autoencoder_10epoch_519.pth")
AE_model.load_state_dict(state_dict)

Swin_trans_model = create_model('swin_base_patch4_window7_224', num_classes=5, pretrained=False, checkpoint_path="/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/swin_transformer_20240421-073851.pth") 


SimCLR_model = ResNetSimCLR(base_model='resnet50', out_dim=4096)
check_points = torch.load("/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/runs/Apr20_09-27-01_server-b/checkpoint_1000.pth.tar")
SimCLR_model.load_state_dict(check_points['state_dict'])

train_model = SimpleViT(
    image_size = 32,
    patch_size = 2,
    num_classes = 5,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels= 57
)

state_dict=torch.load("/mnt/sda1/home/kailexiangzi/SimCLR/SimCLR-master/last_models_addval/final_model_20240426-204138_399.pth")
train_model.load_state_dict(state_dict)
# print(train_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_model.to(device)
AE_model.to(device)
Swin_trans_model.to(device)
SimCLR_model.to(device)


# with torch.no_grad():
#     for image1,image2,image3,labels in valid_loader:
#         # image1=image1.to(device)
#         # image2=image2.to(device)
#         # image3=image3.to(device)
#         feature_1 = AE_model(image2)
#         feature_2 = Swin_trans_model.forward_features(image1)
#         feature_3 = SimCLR_model(image3)
#         feature_1 = torch.reshape(feature_1,(valid_loader_len,-1,32,32))
#         feature_2 = torch.reshape(feature_2,(valid_loader_len,-1,32,32))
#         feature_3 = torch.reshape(feature_3,(valid_loader_len,-1,32,32))

#         feature_all = torch.cat((feature_1,feature_2,feature_3),dim=1).to(device)
#         labels = labels.to(device)

#         outputs = train_model(feature_all)


#         _, preds = torch.max(outputs, 1)
       
#         pred = outputs.detach().cpu().numpy()
#         labels = labels.detach().cpu().numpy()
#         pred = np.argmax(pred,axis=1)
#         metrics = classification_metrics(labels,pred)
#         print("qwk:",metrics["qwk"])
#         print("f1:",metrics["f1"])
#         print("spe:",metrics["spe"])
total = 0
correct = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for image1,image2,image3,labels in valid_loader:
        image1=image1.to(device)
        image2=image2.to(device)
        image3=image3.to(device)
        # predictions = torch.zeros(image1.size(0), 5).to(device)
        labels = labels.to(device)
        # augmentation = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(10),
        #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        # ])


        for _ in range(1):
            # augmented_images = augmentation(image1)
            feature_1 = AE_model(image2)
 
            feature_2 = Swin_trans_model.forward_features(image1)
            feature_3 = SimCLR_model(image3)
            feature_1 = torch.reshape(feature_1,(image1.size(0),-1,32,32))
            feature_2 = torch.reshape(feature_2,(image1.size(0),-1,32,32))
            feature_3 = torch.reshape(feature_3,(image1.size(0),-1,32,32))
        
            feature_all = torch.cat((feature_1,feature_2,feature_3),dim=1).to(device)
            outputs = train_model(feature_all)
            # predictions += outputs
        # predictions /= 5
        predictions = outputs
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
metrics = classification_metrics(np.array(all_labels), np.array(all_preds))
print(f'Accuracy with TTA: {accuracy:.2f}%')
print(f'QWK: {metrics["qwk"]:.4f}')
print(f'F1 Score: {metrics["f1"]:.4f}')
print(f'Specificity: {metrics["spe"]:.4f}')
      

       








