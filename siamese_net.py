import torch
import torch.nn as nn

## to musi być w głównym folderze bo inaczej jest błąd w wczytywaniu modelu w deepsort
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        #Outputs batch X 512 X 1 X 1
        self.net = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=0.4),

            nn.Conv2d(32,64,kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            #nn.Dropout2d(p=0.4),

            nn.Conv2d(64,128,kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            #nn.Dropout2d(p=0.4),


            nn.Conv2d(128,256,kernel_size=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            #nn.Dropout2d(p=0.4),

            nn.Conv2d(256,256,kernel_size=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            #nn.Dropout2d(p=0.4),

            nn.Conv2d(256,512,kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            #1X1 filters to increase dimensions
            nn.Conv2d(512,1024,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),

            )

    def forward_once(self, x):
        output = self.net(x)
        # output = output.view(output.size()[0], -1)
        # output = self.fc(output)

        output = torch.squeeze(output)
        return output

    def forward(self, input1, input2, input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if input3 is not None:
            output3 = self.forward_once(input3)
            return output1, output2, output3

        return output1, output2