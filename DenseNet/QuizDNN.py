import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class quizDNN(nn.Module):
    def __init__(self):
        super(quizDNN, self).__init__()

        self.conv02 = nn.Conv2d(3, 16, 3, bias=False, padding=1)        
        self.batch02 = nn.BatchNorm2d(num_features=16)   

        self.skip_conv3 = nn.Conv2d(19, 32, 3, bias=False, padding=1)
        self.batch_skip_conv3 = nn.BatchNorm2d(num_features=32) 
        
        self.pool04 = nn.MaxPool2d(2, 2)  

        self.conv05 = nn.Conv2d(51, 64, 3, bias=False, padding=1)
        self.batch05 = nn.BatchNorm2d(num_features=64)  

        self.conv06 = nn.Conv2d(115, 128, 3, bias=False, padding=1)
        self.batch06 = nn.BatchNorm2d(num_features=128)

        self.conv07 = nn.Conv2d(243, 256, 3, bias=False, padding=1)
        self.batch07 = nn.BatchNorm2d(num_features=256)

        self.pool08 = nn.MaxPool2d(2, 2) 

        self.conv09 = nn.Conv2d(448, 512, 3, bias=False, padding=1)
        self.batch09 = nn.BatchNorm2d(num_features=512)

        self.conv10 = nn.Conv2d(960, 512, 3, bias=False, padding=1)
        self.batch10 = nn.BatchNorm2d(num_features=512)

        self.conv11 = nn.Conv2d(1472, 512, 3, bias=False, padding=1)
        self.batch11 = nn.BatchNorm2d(num_features=512)

        self.avg_pool = nn.AvgPool2d(kernel_size=8)

        self.conv_FC = nn.Conv2d(512, 10, 1, bias=False, padding=0)

    def forward(self, x):
        x1 = x                                                                            # 3

        x2 = self.batch02(F.relu(self.conv02(x)))                                         # 16

        x3 = self.batch_skip_conv3(F.relu(self.skip_conv3(torch.cat((x1, x2), dim= 1))))  # 32

        x4 = self.pool04((torch.cat((x1, x2, x3), dim= 1)))                               # 51  -- 16
        
        x5 = self.batch05(F.relu(self.conv05(x4)))                                        # 64

        x6 = self.batch06(F.relu(self.conv06(torch.cat((x4, x5), dim= 1))))               # 128

        x7 = self.batch07(F.relu(self.conv07(torch.cat((x4, x5, x6), dim= 1))))           # 256

        x8 = self.pool08((torch.cat((x5, x6, x7), dim= 1)))                               # 448  -- 8

        x9 = self.batch09(F.relu(self.conv09(x8)))                                         # 512

        x10 = self.batch10(F.relu(self.conv10(torch.cat((x8, x9), dim= 1))))               # 512

        x11 = self.batch11(F.relu(self.conv11(torch.cat((x8, x9, x10), dim= 1))))          # 512

        x12 = self.avg_pool(x11)

        x13 = self.conv_FC(x12)

        x14 = x13.view(-1, 10) 

        return F.log_softmax(x14)

def isCudaAvailable():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def displayModelSummary(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    summary(model, input_size=(3, 32, 32))

def quizDNNFileVersion():
    print ('File Version = 1.14 - Used to see, if my changes are reflecting or not')