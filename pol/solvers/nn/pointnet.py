import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel, use_bn=False):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
            self.bn4 = nn.Identity()
            self.bn5 = nn.Identity()

    def forward(self, x):
        '''
        Args: x - BxCxN
        Return: Bx3x3
        '''
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(3).to(x).view(1, 9).repeat(
            batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, *, channel=3, latent_dim=1024, use_bn=True):
        super().__init__()
        # self.stn = STN3d(channel, use_bn=use_bn)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, latent_dim, 1)
        if use_bn:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(latent_dim)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
        self.latent_dim = latent_dim

    def forward(self, x):
        '''
        Args: x - BxNxC
        Return: BxL, L=latent_dim
        '''
        B, N, D = x.size()
        # trans = self.stn(torch.transpose(x, 1, 2))
        # if D > 3:
        #     feature = x[:, :, 3:]
        #     x = x[:, :, :3]
        # x = torch.bmm(x, trans)
        # if D > 3:
        #     x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)  # BxDxN
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_dim)
        return x
