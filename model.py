import torch
from torch_geometric.nn import GCNConv
from torch import nn
from torch_geometric.nn.inits import uniform

from attention import Attention
from torch.nn import Parameter



class Normolization(torch.nn.Module):
    def __init__(self, in_channels):
        super(Normolization, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, data):
        x = data.shape[0]
        y = data.shape[1]

        data_x = data.reshape(1, x, 1, y)
        data_y = torch.relu(self.bn((data_x)))
        output = data_y.reshape(x, y)

        return output


class mymodel(nn.Module):
    def __init__(self, args):
        super(mymodel, self).__init__()
        self.args = args

        self.gcn_lsm1 = GCNConv(self.args.fl, self.args.fl)  # define the first GCN layer to tackle lncRNA similarity
        self.bn1_x = Normolization(self.args.lncRNA_number)
        self.gcn_lsm2 = GCNConv(self.args.fl, self.args.fl)  # define the first GCN layer to tackle lncRNA similarity
        self.bn2_x = Normolization(self.args.lncRNA_number)

        self.gcn_dsm1 = GCNConv(self.args.fd, self.args.fd)  # define the second GCN layer to tackle disease similarity
        self.bn1_y = Normolization(self.args.disease_number)
        self.gcn_dsm2 = GCNConv(self.args.fd, self.args.fd)  # define the second GCN layer to tackle disease similarity
        self.bn2_y = Normolization(self.args.disease_number)

        self.weight = Parameter(torch.Tensor(self.args.fl, self.args.fd))
        self.reset_parameters()
        self.attention_x = Attention(self.args.fl)
        self.attention_y = Attention(self.args.fd)

        self.bn_x = Normolization(self.args.lncRNA_number)
        self.bn_y = Normolization(self.args.disease_number)

    def reset_parameters(self):
        uniform(self.args.fl, self.weight)

    def forward(self, data):
        torch.manual_seed(3407)
        self.x_l = torch.randn(self.args.lncRNA_number, self.args.fl)
        self.x_d = torch.randn(self.args.disease_number, self.args.fd)
        x_l_lsm1 = (self.gcn_lsm1(self.x_l.cuda(), data['LSM']['edges'].cuda(), data['LSM']['data_matrix'][
            data['LSM']['edges'][0], data['LSM']['edges'][1]].cuda()))
        x_l_lsm1 = torch.relu(self.bn1_x(x_l_lsm1))

        x_l_lsm2 = (self.gcn_lsm2(x_l_lsm1.cuda(), data['LSM']['edges'].cuda(), data['LSM']['data_matrix'][
            data['LSM']['edges'][0], data['LSM']['edges'][1]].cuda()))
        x_l_lsm2_n = self.bn2_x(x_l_lsm2)
        x_l_lsm2 = torch.relu(x_l_lsm1 + x_l_lsm2_n)

        y_d_dsm1 = (self.gcn_dsm1(self.x_d.cuda(), data['DSM']['edges'].cuda(), data['DSM']['data_matrix'][
            data['DSM']['edges'][0], data['DSM']['edges'][1]].cuda()))
        y_d_dsm1 = torch.relu(self.bn1_y(y_d_dsm1))

        y_d_dsm2 = (self.gcn_dsm2(y_d_dsm1, data['DSM']['edges'].cuda(), data['DSM']['data_matrix'][
            data['DSM']['edges'][0], data['DSM']['edges'][1]].cuda()))
        y_d_dsm2_n = self.bn2_y(y_d_dsm2)
        y_d_dsm2 = torch.relu(y_d_dsm1 + y_d_dsm2_n)

        x = (self.attention_x(x_l_lsm2))
        y = (self.attention_y(y_d_dsm2))

        x = torch.relu(self.bn_x(x))
        y = torch.relu(self.bn_y(y))

        temp_o = torch.matmul(x, self.weight)
        output = torch.matmul(temp_o, torch.t(y))
        out = torch.sigmoid(output)
        return x, y, out