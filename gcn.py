import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, global_mean_pool, global_add_pool

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, layer):
        super(GCNEncoder, self).__init__()
        if layer == "gcn":
            # linear->GCN layer=2
            self.lin = nn.Linear(in_channels, out_channels)
            self.dropout1 = nn.Dropout()
            self.conv1 = GCNConv(out_channels, out_channels, improved=True) 
            self.dropout2 = nn.Dropout()
            self.conv2 = GCNConv(out_channels, out_channels, improved=True)
        elif layer == "graphsage":
            # GraphSAGE
            self.lin = nn.Linear(in_channels, out_channels)
            self.dropout1 = nn.Dropout()
            self.conv1 = SAGEConv(out_channels, out_channels)
            self.dropout2 = nn.Dropout()
            self.conv2 = SAGEConv(out_channels, out_channels)
        elif layer == "gat":       
            # GAT
            self.lin = nn.Linear(in_channels, out_channels)
            self.dropout1 = nn.Dropout()
            self.conv1 = GATConv(out_channels, out_channels, heads=1)
            self.dropout2 = nn.Dropout()
            self.conv2 = GATConv(out_channels, out_channels, heads=1)  
        elif layer == "gin":
            # GIN
            self.lin = nn.Linear(in_channels, out_channels)
            self.dropout1 = nn.Dropout()
            self.conv1 = GINConv(
                            nn.Sequential(
                                nn.Linear(out_channels, out_channels),
                                nn.ReLU(),
                                nn.Linear(out_channels, out_channels)
                            )
                         )
            self.dropout2 = nn.Dropout()
            self.conv2 = GINConv(
                            nn.Sequential(
                                nn.Linear(out_channels, out_channels),
                                nn.ReLU(),
                                nn.Linear(out_channels, out_channels)
                            )
                         )
    
    def forward(self, x, edge_index):
        out = self.lin(x)
        out = self.dropout1(out)
        identity1 = out
        out1 = self.conv1(out, edge_index).relu()
        out1 += identity1
        out = self.dropout2(out1)
        identity2 = out
        out2 = self.conv2(out, edge_index)
        out2 += identity2
        return [out1, out2]


class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, nclass, ncov):
        super(GCNClassifier, self).__init__()
        self.ncov = ncov
        self.gcnencoder = GCNEncoder(in_channels, hidden_channels)
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(hidden_channels, 8)
        self.linear2 = nn.Linear(8+self.ncov, nclass)
        
    def forward(self, x, edge_index, cov, batch):
        outs = self.gcnencoder(x, edge_index)
        
        out = global_mean_pool(outs[-1], batch)
#         out = global_add_pool(outs[-1], batch)
#         out = torch.hstack([global_mean_pool(out, batch) for out in outs])
        
        out = self.dropout(out)
        out = self.linear1(out).relu()
        out = torch.cat((out, cov.view(-1, self.ncov)), dim=1)
        out = self.linear2(out)
        return out
    

from sklearn.utils import shuffle
from torch.autograd import Variable

class LSTMAttention(nn.Module):
    def __init__(self, hidden_channels, num_layers, nclass, attention, device):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_channels
        self.device = device
        self.num_layers = num_layers
        self.attention = attention
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True, num_layers=self.num_layers, dropout=0.5, bidirectional=False)
        self.w0 = nn.Linear(hidden_channels, hidden_channels)
        
        if self.attention == "attention":
            self.attn_fc = nn.Linear(hidden_channels*2, 1)

        elif self.attention == "selfattention":
            # multi-head self attention
            self.heads = 1
            self.attn_fc = nn.ModuleList([nn.Linear(hidden_channels, 1) for h in range(self.heads)])
        
    def init_hidden(self,batch_size):
        h0 = Variable(torch.randn(self.num_layers, batch_size, self.hidden_dim)).to(self.device)
        c0 = Variable(torch.randn(self.num_layers, batch_size, self.hidden_dim)).to(self.device)
        return (h0, c0)


    def calAttention(self, rnn_out, last_state):
        # rnn_out     (batch, seq_len, hidden_dim) 
        # last_state  (1, batch, hidden_dim)
        
        # attention
        last_state = last_state.squeeze(dim=0).unsqueeze(dim=1)
        last_state = last_state.tile((1,rnn_out.size(1),1))
        merged_state = torch.cat((rnn_out, last_state), dim=2)
        weights = self.attn_fc(merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(dim=2), dim=1)
        return weights, torch.bmm(torch.transpose(rnn_out, 1, 2), weights.unsqueeze(dim=2)).squeeze(dim=2)

    
    def calSelfAttention(self, rnn_out, last_state):
        # rnn_out     (batch, seq_len, hidden_dim) 
        # last_state  (1, batch, hidden_dim)

        # multi-head self-attention
        heads_weights = torch.zeros((rnn_out.size(0), rnn_out.size(1))).to(self.device)
        for h in range(self.heads):
            weights = self.attn_fc[h](rnn_out)
            weights = torch.nn.functional.softmax(weights.squeeze(dim=2), dim=1)
            heads_weights += weights
        heads_weights /= self.heads
        return heads_weights, torch.bmm(torch.transpose(rnn_out, 1, 2), heads_weights.unsqueeze(dim=2)).squeeze(dim=2)

    def forward(self, X):        
        hidden= self.init_hidden(X.size()[0])
        rnn_out, hidden = self.lstm(X, hidden)
        h_n, c_n = hidden
        if self.attention == "attention":
            rnn_out = self.w0(rnn_out).tanh()
            attn, out = self.calAttention(rnn_out, h_n)
        elif self.attention == "selfattention":
            rnn_out = self.w0(rnn_out).tanh()
            attn, out = self.calSelfAttention(rnn_out, h_n)
        else:
            out = rnn_out[:,-1,:]
            attn = torch.zeros((X.size(0), X.size(1)))
        return attn, out
    
class temporalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, nclass, ncov, layer, attention, device):
        super(temporalGNN, self).__init__()
        self.ncov = ncov
        self.in_channels = in_channels
        self.gcnencoder = GCNEncoder(in_channels, hidden_channels, layer)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.lstmattention = LSTMAttention(hidden_channels, 1, nclass, attention, device)
        self.linear1 = nn.Linear(hidden_channels, 8)
        self.linear2 = nn.Linear(8+self.ncov, nclass)
        
    def forward(self, x, edge_index, cov, batch):
        x1 = x[:,:self.in_channels]
        x2 = x[:,self.in_channels:self.in_channels*2]
        x3 = x[:,self.in_channels*2:]
        outs1 = self.gcnencoder(x1, edge_index)
        outs2 = self.gcnencoder(x2, edge_index)
        outs3 = self.gcnencoder(x3, edge_index)
        
        out1 = global_mean_pool(outs1[-1], batch)
        out2 = global_mean_pool(outs2[-1], batch)
        out3 = global_mean_pool(outs3[-1], batch)

        # (N,L,H)
        out = torch.cat((out1.view(out1.size(0),1,out1.size(1)), 
                         out2.view(out2.size(0),1,out2.size(1)), 
                         out3.view(out3.size(0),1,out3.size(1))), dim=1)
        out = self.dropout1(out)
        attn, out = self.lstmattention(out)
        out = self.dropout2(out)
        out = self.linear1(out).relu()
        out = torch.cat((out, cov.view(-1, self.ncov)), dim=1)
        out = self.linear2(out)
        return attn, out