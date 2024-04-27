import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from dataloader import train_loader, valid_loader
import datetime
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=5) 
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def train_model(model, criterion, optimizer, num_epochs=25):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        return model

    
    trained_model = train_model(model, criterion, optimizer, num_epochs=10)

    
    model_dir = 'model_saved'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    
    model_filename = os.path.join(model_dir, f'swin_transformer_{start_time}.pth')
    torch.save(trained_model.state_dict(), model_filename)

if __name__ == '__main__':
    main()
