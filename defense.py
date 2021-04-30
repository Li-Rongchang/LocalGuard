import ite
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Optim_noise(nn.Module):
    def __init__(self, x_dim, y_dim):  # feature0 = 3, feature1 = 6
        super(Optim_noise, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.x_dim, 100),
            # nn.ReLU(True),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 100),
            # nn.ReLU(True),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(100, self.y_dim),
            # nn.ReLU(),

            )

    def forward(self, X):
        if X.dim() > 2:
            X = X.view(X.size(0), -1)
        out = self.mlp(X)
        # print(out)

        return out






from torch.autograd import Variable


def ml_inf2(emb,labels,ee,st):
    # a = e
    # print('as')

    # if (ee == 1 or ee == 10 or ee == 20 or ee == 25 or ee == 27 or ee == 29):
    # if (ee > 0) and (st == 'train'):
    if (ee==1 or ee%15==0) and (st=='train'):
        raw_emb = emb.clone()
        noise_num = emb.shape[0] * emb.shape[1]
        raw_emb = Variable(raw_emb, requires_grad=True)

        model = Optim_noise(noise_num, noise_num).cuda()
        bata1 = torch.ones(noise_num).float().cuda()
        # print(emb.shape[0]*emb.shape[1])

        emb = Variable(emb, requires_grad=True)

        optimizer = torch.optim.Adam([
            dict(params=model.parameters()),
        ], lr=0.01,weight_decay=1e-5)

        import random

        lam = 1
        a = []
        leibie = 6
        for j in range(emb.shape[0]//leibie):
            for i in range(leibie):
                a.append(i)
        # print(len(a))
        while len(a)<emb.shape[0]:
            a.append(0)
        # print(a.shape)

        random.shuffle(a)

        c = torch.tensor(a).reshape(emb.shape[0]).cuda()

        for e in range(50):

            optimizer.zero_grad()
            model.train()

            injection_noise = model(bata1).reshape(emb.shape)

            new_emb = injection_noise + emb #0.5*F.l1_loss(new_emb,raw_emb) +F.l1_loss(new_emb,raw_emb) +

            # loss1 = nn.MSELoss()(new_emb,raw_emb) +lam*nn.CrossEntropyLoss()(new_emb,c) #nn.KLDivLoss()(new_emb,raw_emb)
            loss2 = nn.MSELoss()(new_emb,raw_emb)
            loss3 = nn.CrossEntropyLoss()(new_emb,c)
            # loss3 = F.nll_loss(nn.Sigmoid()(new_emb),c)
            loss1 = 1*loss2 + lam*loss3
            loss1.backward(retain_graph=True)
            # print([x.grad for x in optimizer.param_groups[0]['params']])
            optimizer.step()
            print(loss1)

        # injection_noise = 0.5*torch.rand(emb.shape).cuda()
        # injection_noise = torch.zeros(emb.shape).cuda()
        np.save('./noise.npy',injection_noise.cpu().detach().numpy())
        # print('youhua',injection_noise)


    else:
        injection_noise = torch.from_numpy(np.load('./noise.npy')).cuda()
        # print('wuyouhua',injection_noise)
        # a=a
    # injection_noise = torch.zeros(emb.shape).cuda()


    return injection_noise




# s=ml_inf2(torch.randn([200,200]).cuda(),torch.ones([200,1]).cuda())
# print(s)
