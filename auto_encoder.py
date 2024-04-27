import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder1 = nn.Conv2d(3, 128, kernel_size=5, stride=1,
                                  padding=2)
        self.encoder2 = nn.Conv2d(128, 128, kernel_size=5, stride=1,
                                  padding=2)
        self.encoder3 = nn.Conv2d(128, 128, kernel_size=5, stride=1,
                                  padding=2)
        self.encoder4 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                                  padding=1)
        self.encoder5 = nn.Conv2d(128, 256, kernel_size=3, stride=1,
                                  padding=1)
        self.encoder6 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                  padding=1)
        self.encoder7 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                  padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=4)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.decoder1 = nn.Conv2d(128, 3, kernel_size=5, stride=1,
                                  padding=2)
        self.decoder2 = nn.Conv2d(128, 128, kernel_size=5, stride=1,
                                  padding=2)
        self.decoder3 = nn.Conv2d(128, 128, kernel_size=5, stride=1,
                                  padding=2)
        self.decoder4 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                                  padding=1)
        self.decoder5 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                                  padding=1)
        self.decoder6 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                  padding=1)
        self.decoder7 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                  padding=1)

    def encode1(self, x):
        x = F.leaky_relu(self.encoder1(x))
        x = F.leaky_relu(self.encoder2(x))
        x = self.maxpool1(x)
        return x

    def decode1(self, x):
        x = self.upsample1(x)
        x = F.sigmoid(self.decoder2(x))
        x = F.sigmoid(self.decoder1(x))
        return x

    def encode2(self, x):
        x = F.leaky_relu(self.encoder3(x))
        x = F.leaky_relu(self.encoder4(x))
        x = self.maxpool2(x)
        return x

    def decode2(self, x):
        x = self.upsample2(x)
        x = F.leaky_relu(self.decoder4(x))
        x = F.leaky_relu(self.decoder3(x))
        return x

    def encode3(self, x):
        x = F.leaky_relu(self.encoder5(x))
        x = F.leaky_relu(self.encoder6(x))
        x = self.maxpool1(x)
        x = F.leaky_relu(self.encoder7(x))
        return x

    def decode3(self, x):
        x = F.leaky_relu(self.decoder7(x))
        x = self.upsample1(x)
        x = F.leaky_relu(self.decoder6(x))
        x = F.leaky_relu(self.decoder5(x))
        return x

    def encode(self, x):
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        return x

    def decode(self, x):
        x = self.decode3(x)
        x = self.decode2(x)
        x = self.decode1(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        # x = self.decode(x)
        return x


    # if load:
    #     state_dict = torch.load("/kaggle/working/autoencoder.pth")
    #     autoencoder.load_state_dict(state_dict)