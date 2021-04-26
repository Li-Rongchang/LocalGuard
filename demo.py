import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.utils import train_test_split_edges
import argparse
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import json
from torch.nn import Linear



class Net_model(torch.nn.Module):
    def __init__(self, name='GCNConv'):
        super(Net_model, self).__init__()
        self.name = name
        self.mlp_attack = Linear(64, dataset.num_classes)

    def forward(self,x2):
        att = self.mlp_attack(x2)
        return F.log_softmax(att, dim=1)


class Net(torch.nn.Module):
    def __init__(self, name='GCNConv'):
        super(Net, self).__init__()
        self.name = name
        if (name == 'GCNConv'):
            self.conv1 = GCNConv(data.x_A.shape[1], 128)
            self.conv2 = GCNConv(128, 64)
        elif (name == 'ChebConv'):
            self.conv1 = ChebConv(data.x_A.shape[1], 128, K=2)
            self.conv2 = ChebConv(128, 64, K=2)
        elif (name == 'GATConv'):
            self.conv1 = GATConv(data.x_A.shape[1], 128)
            self.conv2 = GATConv(128, 64)

        self.mlp = Linear(128, 64)
        self.mlp_attack = Linear(64,dataset.num_classes)

    def forward(self, pos_edge_index, neg_edge_index,e,st):

        x1 = F.relu(self.conv1(data.x_A, data.train_pos_edge_index))
        # x1 = self.conv1(data.x_A, data.train_pos_edge_index)
        x1 = self.conv2(x1, data.train_pos_edge_index)
        x2 = F.relu(self.conv1(data.x_B, data.train_pos_edge_index))
        # x2 = self.conv1(data.x_B, data.train_pos_edge_index)
        x2 = self.conv2(x2, data.train_pos_edge_index)

        """embedding protection stage"""

        # localguard
        from defense import ml_inf2
        noise = ml_inf2(x2,labels,e,st)
        noise = torch.tensor(noise,requires_grad=True)

        li = 1    # default 1

        x2 = x2+li*noise.cuda()


            # print(noise)

            # x1 = torch.rand(x1.shape).cuda() + x1
            # import ite
            # co = ite.cost.MIShannon_DKL()
            # a = x2.cpu().detach().numpy()
            # d = labels.cpu().detach().numpy().reshape((labels.shape[0], 1))
            # # print(d.shape)
            # # print(a.shape)
            # ds = [x2.shape[1], 1]
            # y = np.concatenate((a, d), axis=1)
            #
            # i = co.estimation(y, ds)
            # print('ml', i)


        x = self.mlp(torch.cat((x1, x2), dim=1))



        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        att = self.mlp_attack(x2)

        if e % 10 == 0:
            np.save('./{}_epoch{}_emb2.npy'.format(args.dataset,e), x2.cpu().detach().numpy())


        res = torch.einsum("ef,ef->e", x_i, x_j)

        return res, F.log_softmax(att, dim=1) #, F.log_softmax(attr, dim=1)


parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=30, help='random seed')
parser.add_argument('--lambda_reg', type=float, default=0, help='Regularization strength for gradient ascent-descent')
parser.add_argument('--gnn_layers', type=int, default=2, help='Layers of GNN')
parser.add_argument('--gnn_types', type=str, default='ChebConv', help='Types of GNN')
parser.add_argument('--finetune_epochs', type=int, default=100, help='Finetune epochs')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')

args = parser.parse_args()

res = {}
for seed in [100]:
    for l in [0]:
        for m in ['GCNConv']:#'GATConv','ChebConv'

            args.seed = seed
            args.lambda_reg = l

            try:
                res[m][seed][l] = {}
            except:
                try:
                    res[m][seed] = {l: {}}
                except:
                    res[m] = {seed: {l: {}}}

            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(args.seed)

            dataset = args.dataset
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)

            dataset = Planetoid(path, dataset, T.NormalizeFeatures())
            data = dataset[0]


            if args.use_gdc:
                gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                            normalization_out='col',
                            diffusion_kwargs=dict(method='ppr', alpha=0.05),
                            sparsification_kwargs=dict(method='topk', k=128,
                                                       dim=0), exact=True)
                data = gdc(data)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            labels = data.y.to(device)
            edge_index, edge_weight = data.edge_index.to(device), data.edge_attr


            print(labels.size())
            # Train/validation/test
            data = train_test_split_edges(data, val_ratio=0.4, test_ratio=0.2)


            print(labels)


            data.x_A = torch.split(data.x, data.x.size()[1] // 2, dim=1)[0].to(device)
            data.x_B = torch.split(data.x, data.x.size()[1] // 2, dim=1)[1].to(device)
            # data.x_A = torch.zeros(torch.split(data.x, data.x.size()[1] // 2, dim=1)[0].shape).to(device)
            # data.x_B = torch.zeros(torch.split(data.x, data.x.size()[1] // 2, dim=1)[1].shape).to(device)
            data.train_A_pos_edge_index = data.train_pos_edge_index # default
            data.train_B_pos_edge_index = data.train_pos_edge_index # for citeseer dataset
            # print(type(data))

            model, data = Net(m).to(device), data.to(device)
            model_attack = Net_model().to(device)
            optim_attack = torch.optim.Adam([
                    dict(params=model_attack.parameters(), weight_decay=0)
                ], lr=args.lr)




            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=0),
                dict(params=model.conv2.parameters(), weight_decay=0),
                dict(params=model.mlp.parameters(), weight_decay=0)
            ], lr=args.lr)


            optimizer_att = torch.optim.Adam([
                dict(params=model.mlp_attack.parameters(), weight_decay=5e-4),
                # dict(params=model.attack.parameters(), weight_decay=5e-4),
            ], lr=args.lr)


            def get_link_labels(pos_edge_index, neg_edge_index):
                link_labels = torch.zeros(pos_edge_index.size(1) +
                                          neg_edge_index.size(1)).float().to(device)
                link_labels[:pos_edge_index.size(1)] = 1.
                return link_labels



            def train(e,st):
                model.train()
                optimizer.zero_grad()

                x, pos_edge_index = data.x, data.train_pos_edge_index
                # x is the feature, pos_edge_index is the postive node pair

                _edge_index, _ = remove_self_loops(pos_edge_index)
                pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                                   num_nodes=x.size(0))

                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
                    num_neg_samples=pos_edge_index.size(1))

                link_logits = model(pos_edge_index, neg_edge_index,e,st)[0]
                link_labels = get_link_labels(pos_edge_index, neg_edge_index)

                loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
                loss.backward()
                optimizer.step()

                return loss


            def test(e,st):
                model.eval()
                perfs = []
                for prefix in ["val", "test"]:
                    pos_edge_index, neg_edge_index = [
                        index for _, index in data("{}_pos_edge_index".format(prefix),
                                                   "{}_neg_edge_index".format(prefix))
                    ]
                    link_probs = torch.sigmoid(model(pos_edge_index, neg_edge_index,e,st)[0])
                    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
                    link_probs = link_probs.detach().cpu().numpy()
                    link_labels = link_labels.detach().cpu().numpy()

                    perfs.append(roc_auc_score(link_labels, link_probs))
                return perfs


            best_val_perf = test_perf = 0
            for epoch in range(1, args.num_epochs + 1):
                train_loss = train(epoch,'train')
                val_perf, tmp_test_perf = test(epoch,'notrain')
                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    test_perf = tmp_test_perf
                    res[m][seed][l]['task'] = { 'test': test_perf}
                log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, train_loss, val_perf, tmp_test_perf))



            def train_attack(x2):
                model_attack.train()
                optim_attack.zero_grad()
                F.nll_loss(model_attack(x2)[data.train_mask],
                           labels[data.train_mask]).backward()
                optim_attack.step()

            @torch.no_grad()
            def test_attack(x2):
                model_attack.eval()
                accs = []
                m = ['train_mask', 'val_mask', 'test_mask']
                i = 0
                for _, mask in data('train_mask', 'val_mask', 'test_mask'):

                    logits = model_attack(x2)

                    pred = logits[mask].max(1)[1]

                    macro = accuracy_score((data.y[mask]).cpu().numpy(), pred.cpu().numpy())
                    accs.append(macro)

                    i += 1
                return accs


            # best_val_acc = test_acc = 0
            # for epoch in range(1, args.finetune_epochs + 1):
            #     x2 = torch.from_numpy(np.load('./epoch10_emb2.npy')).cuda()
            #     train_attack(x2)
            #     train_acc, val_acc, tmp_test_acc = test_attack(x2)
            #     if val_acc > best_val_acc:
            #         best_val_acc = val_acc
            #         test_acc = tmp_test_acc
            #         res[m][seed][l]['attack'] = {'test': test_acc}
            #     log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            #     print(log.format(epoch, train_acc, val_acc, tmp_test_acc))

            from metric import classifier, ml_privacy, classifier_other
            classifier(args, data.y.cpu().detach().numpy())
            # ml_privacy(args, data.y.cpu().detach().numpy())
            # classifier_other(args, data.y.cpu().detach().numpy())




print(res)
