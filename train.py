import numpy as np
import torch
from modules import transform, resnet, network, contrastive_loss
from torch.nn.functional import normalize

def train_net(net, data_loader, optimizer, batch_size, zeta):

    net.train()
    for param in net.parameters():
        param.requires_grad = True

    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()

        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        h_i = net.resnet(x_i)
        h_j = net.resnet(x_j)

        z_i = normalize(net.instance_projector(h_i), dim=1)
        z_j = normalize(net.instance_projector(h_j), dim=1)

        loss = contrastive_loss.C3_loss(z_i, z_j, batch_size, zeta)
        loss.backward()
        optimizer.step()

    return net , optimizer
