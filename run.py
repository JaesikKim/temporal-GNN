from absl import app, flags, logging

import pandas as pd
import numpy as np
import scipy.io
import networkx
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch_geometric
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from gcn import temporalGNN

import copy

flags.DEFINE_string('input_pth', 'data/freesurfer_all_JK.csv', '')
flags.DEFINE_string('labels_loc', 'data/label_new.xlsx', '')
flags.DEFINE_string('topology_pth', 'data/brain-network.mat', '')
flags.DEFINE_string('cuda', 'cuda:0', '')
flags.DEFINE_string('layer', 'gcn', '') # gcn, graphsage, gat, gin
flags.DEFINE_string('attention', 'vanilla', '') # vanilla, attention, selfattention
flags.DEFINE_integer('repeats', 60, '')
flags.DEFINE_integer('epochs', 50, '')
flags.DEFINE_integer('early_stopping_tol', 10, '')
flags.DEFINE_integer('hidden_channels', 16, '')
flags.DEFINE_integer('num_features', 4, '')
flags.DEFINE_integer('nclass', 3, '')
flags.DEFINE_float('lr', 5e-4, '')
FLAGS = flags.FLAGS

def load_data():
    
    data = pd.read_csv(FLAGS.input_pth, skipinitialspace=True)
    data_bl = data[data.VISCODE=='bl']
    data_m06 = data[data.VISCODE=='m06']
    data_m12 = data[data.VISCODE=='m12']
    data_m24 = data[data.VISCODE=='m24']

    y = data_m24.pop('DXCURRENT').values

    print('CN: ' + str(y[y=='CN'].shape[0]) + ' samples')
    print('MCI: ' + str(y[y=='MCI'].shape[0]) + ' samples')
    print('AD: ' + str(y[y=='AD'].shape[0]) + ' samples')
    print('Total: ' + str(y.shape[0]) + ' samples' )

    y[y=='CN'] = 0
    y[y=='MCI'] = 1
    y[y=='AD'] = 2
    y = np.array(y, dtype='float64')

    xls = pd.ExcelFile(FLAGS.labels_loc)

    labels_ta = pd.read_excel(xls, 'Thickness.Avg').pop('FLDNAME').values
    labels_ts = pd.read_excel(xls, 'Thickness.Std').pop('FLDNAME').values
    labels_vol = pd.read_excel(xls, 'Volume').pop('FLDNAME').values
    labels_area = pd.read_excel(xls, 'Area').pop('FLDNAME').values

    fs_avg_ct_bl = data_bl[labels_ta]
    fs_std_ct_bl = data_bl[labels_ts]
    fs_surf_bl = data_bl[labels_area]
    fs_vol_bl = data_bl[labels_vol]

    fs_avg_ct_m06 = data_m06[labels_ta]
    fs_std_ct_m06 = data_m06[labels_ts]
    fs_surf_m06 = data_m06[labels_area]
    fs_vol_m06 = data_m06[labels_vol]
    
    fs_avg_ct_m12 = data_m12[labels_ta]
    fs_std_ct_m12 = data_m12[labels_ts]
    fs_surf_m12 = data_m12[labels_area]
    fs_vol_m12 = data_m12[labels_vol]

    cov = data_bl[['AGE','PTGENDER','Field','PTEDUCAT']]
        
    # input preprocessing (img_dim: num_sub x 68 x 4)
    fs_avg_ct_bl = np.expand_dims(fs_avg_ct_bl,axis=2)
    fs_std_ct_bl = np.expand_dims(fs_std_ct_bl,axis=2)
    fs_surf_bl = np.expand_dims(fs_surf_bl,axis=2)
    fs_vol_bl = np.expand_dims(fs_vol_bl,axis=2)

    fs_avg_ct_m06 = np.expand_dims(fs_avg_ct_m06,axis=2)
    fs_std_ct_m06 = np.expand_dims(fs_std_ct_m06,axis=2)
    fs_surf_m06 = np.expand_dims(fs_surf_m06,axis=2)
    fs_vol_m06 = np.expand_dims(fs_vol_m06,axis=2)
    
    fs_avg_ct_m12 = np.expand_dims(fs_avg_ct_m12,axis=2)
    fs_std_ct_m12 = np.expand_dims(fs_std_ct_m12,axis=2)
    fs_surf_m12 = np.expand_dims(fs_surf_m12,axis=2)
    fs_vol_m12 = np.expand_dims(fs_vol_m12,axis=2)

    img = np.concatenate((fs_avg_ct_bl,fs_std_ct_bl,fs_surf_bl,fs_vol_bl,
                          fs_avg_ct_m06,fs_std_ct_m06,fs_surf_m06,fs_vol_m06,
                          fs_avg_ct_m12,fs_std_ct_m12,fs_surf_m12,fs_vol_m12),axis=2)

    
    network = scipy.io.loadmat(FLAGS.topology_pth)
    topology = network['grp_sconn']
    topology = (topology - topology.min()) / (topology.max() - topology.min())
    # binarize topology
    topology = (topology>0.05).astype(float)
    return img, cov, y, topology

def normalize(img_train, cov_train, label_train, img_val, cov_val, label_val, img_test, cov_test, label_test, topology):

    img_F0_scaler = StandardScaler().fit(img_train.reshape(len(img_train),-1)[cov_train.Field == 0])
    img_F1_scaler = StandardScaler().fit(img_train.reshape(len(img_train),-1)[cov_train.Field == 1])
    
    img_train[cov_train.Field == 0] = img_F0_scaler.transform(img_train.reshape(len(img_train),-1)[cov_train.Field == 0]).reshape(*img_train[cov_train.Field == 0].shape)
    img_train[cov_train.Field == 1] = img_F1_scaler.transform(img_train.reshape(len(img_train),-1)[cov_train.Field == 1]).reshape(*img_train[cov_train.Field == 1].shape)

    img_val[cov_val.Field == 0] = img_F0_scaler.transform(img_val.reshape(len(img_val),-1)[cov_val.Field == 0]).reshape(*img_val[cov_val.Field == 0].shape)
    img_val[cov_val.Field == 1] = img_F1_scaler.transform(img_val.reshape(len(img_val),-1)[cov_val.Field == 1]).reshape(*img_val[cov_val.Field == 1].shape)
    
    img_test[cov_test.Field == 0] = img_F0_scaler.transform(img_test.reshape(len(img_test),-1)[cov_test.Field == 0]).reshape(*img_test[cov_test.Field == 0].shape)
    img_test[cov_test.Field == 1] = img_F1_scaler.transform(img_test.reshape(len(img_test),-1)[cov_test.Field == 1]).reshape(*img_test[cov_test.Field == 1].shape)
    
    cov_scaler = StandardScaler().fit(cov_train)
    cov_train = cov_scaler.transform(cov_train)
    cov_val = cov_scaler.transform(cov_val)
    cov_test = cov_scaler.transform(cov_test)
    
    train_data_list = []
    for i in range(img_train.shape[0]):
        data = torch_geometric.utils.from_networkx(networkx.convert_matrix.from_numpy_matrix(topology))
        data.x = torch.tensor(img_train[i,:,:]).float()
        data.y = torch.tensor(label_train[i]).long()
        data.cov = torch.tensor(cov_train[i,:]).float()
        train_data_list.append(data)
    val_data_list = []
    for i in range(img_val.shape[0]):
        data = torch_geometric.utils.from_networkx(networkx.convert_matrix.from_numpy_matrix(topology))
        data.x = torch.tensor(img_val[i,:,:]).float()
        data.y = torch.tensor(label_val[i]).long()
        data.cov = torch.tensor(cov_val[i,:]).float()
        val_data_list.append(data)
    test_data_list = []
    for i in range(img_test.shape[0]):
        data = torch_geometric.utils.from_networkx(networkx.convert_matrix.from_numpy_matrix(topology))
        data.x = torch.tensor(img_test[i,:,:]).float()
        data.y = torch.tensor(label_test[i]).long()
        data.cov = torch.tensor(cov_test[i,:]).float()
        test_data_list.append(data)
        
    train_loader = DataLoader(train_data_list, batch_size=1, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


def train(train_loader, val_loader, test_loader, model, optimizer, criterion, device):
    best_acc = 0
    best_val_record = None
    best_model = None
    early_stopping_cnt=0
    for epoch in range(1, FLAGS.epochs + 1):
        early_stopping_cnt += 1
        model.train()
        train_loss = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            attn, prob = model(data.x, data.edge_index, data.cov, data.batch)
            loss = criterion(prob, data.y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            corr_cnt = 0
            total_cnt = 0
            val_record = []
            for data in val_loader:
                data = data.to(device)
                attn, prob = model(data.x, data.edge_index, data.cov, data.batch)
                prob = F.softmax(prob, dim=1)
                _, pred = torch.max(prob, dim=1)
                for i in range(len(data.y)):
                    val_record.append(["val"]+prob[i,:].tolist()+[pred.item(), data.y[i].item()])
                    
                corr_cnt += sum(pred == data.y).item()
                total_cnt += len(data.y)
            acc = corr_cnt/total_cnt
            print('Epoch: {:03d}, Train Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, np.mean(train_loss), acc))
        
        if acc > best_acc:
            early_stopping_cnt = 0
            best_acc = acc
            best_model = copy.deepcopy(model)
            best_val_record = val_record
            
            model.eval()
            with torch.no_grad():
                corr_cnt = 0
                total_cnt = 0
                test_record = []
                for data in test_loader:
                    data = data.to(device)
                    attn, prob = model(data.x, data.edge_index, data.cov, data.batch)
                    prob = F.softmax(prob, dim=1)
                    _, pred = torch.max(prob, dim=1)
                    for i in range(len(data.y)):
                        test_record.append(["test"]+prob[i,:].tolist()+[pred.item(), data.y[i].item()])
                    
                    corr_cnt += sum(pred == data.y).item()
                    total_cnt += len(data.y)
                test_acc = corr_cnt/total_cnt
            print('Test Acc: {:.4f}'.format(test_acc))
        if (early_stopping_cnt-1) > FLAGS.early_stopping_tol:
            break
            
    best_val_record = pd.DataFrame(best_val_record, columns=['split', 'prob_0', 'prob_1', 'prob_2', 'pred', 'label'])
    test_record = pd.DataFrame(test_record, columns=['split', 'prob_0', 'prob_1', 'prob_2', 'pred', 'label'])

    return pd.concat((best_val_record, test_record), axis=0)

def accuracy(df):
    df_val = df[df.split == 'val']
    df_test = df[df.split == 'test']
    
    val_acc = (df_val.pred == df_val.label).sum()/len(df_val)
    test_acc = (df_test.pred == df_test.label).sum()/len(df_test)
    return val_acc, test_acc

def main(_):
    img, cov, y, topology = load_data()
    val_accs = []
    test_accs = []
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=FLAGS.repeats, random_state=38)
    for r, (train_index, test_index) in enumerate(rkf.split(img, y)):    
        # split into training, validation, and test data
        img_train, img_test, cov_train, cov_test, label_train, label_test = img[train_index], img[test_index], cov.iloc[train_index], cov.iloc[test_index], y[train_index], y[test_index]
        img_train, img_val, cov_train, cov_val, label_train, label_val = train_test_split(img_train, cov_train, label_train, test_size=0.25, stratify=label_train, random_state=38)
        train_loader, val_loader, test_loader = normalize(img_train, cov_train, label_train, img_val, cov_val, label_val, img_test, cov_test, label_test, topology)

        # move to GPU (if available)
        device = torch.device(FLAGS.cuda if torch.cuda.is_available() else 'cpu')

        # model
        model = temporalGNN(FLAGS.num_features, FLAGS.hidden_channels, FLAGS.nclass, cov.shape[1], FLAGS.layer, FLAGS.attention, device).to(device)    

        # inizialize the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr)
        criterion = nn.CrossEntropyLoss()

        # train the model
        record = train(train_loader, val_loader, test_loader, model, optimizer, criterion, device)
        val_acc, test_acc = accuracy(record)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        
    # save records
    print("Val: {:.3f} {:.3f}".format(np.mean(val_accs), np.std(val_accs)))
    print("Test: {:.3f} {:.3f}".format(np.mean(test_accs), np.std(test_accs)))
if __name__ == '__main__':
    app.run(main)