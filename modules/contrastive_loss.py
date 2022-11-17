import torch


def C3_loss(z_i, z_j, batch_size, zeta):

    z = torch.cat((z_i, z_j), dim=0)
    multiply = torch.matmul(z, z.T)

    a = torch.ones([batch_size])
    mask = 2 * (torch.diag(a, -batch_size) + torch.diag(a, batch_size) + torch.eye(2 * batch_size))
    mask = mask.cuda()

    exp_mul = torch.exp(multiply)
    numerator = torch.sum(torch.where((multiply + mask) > zeta, exp_mul, torch.zeros(multiply.shape).cuda()), dim=1)
    den = torch.sum(exp_mul, dim=1)

    return -torch.sum(torch.log(torch.div(numerator, den))) / batch_size
