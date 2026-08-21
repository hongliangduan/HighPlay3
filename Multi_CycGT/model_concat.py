import os
import json
import pickle
import torch.nn as nn
from models import Transformer_test, Model_TGCN, config
from data_pretreatment import func
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import dgl
from dgllife.utils import *
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor

torch.cuda.empty_cache()
import os, random, numpy as np, torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
except Exception:
    pass


def main_():
    want_cuda = torch.cuda.is_available()
    have_dgl_cuda = False
    if want_cuda:
        try:
            gtest = dgl.graph(([0], [0]))
            gtest = gtest.to('cuda:1')
            have_dgl_cuda = True
        except Exception:
            have_dgl_cuda = False

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    gcn_device = torch.device('cuda:1' if have_dgl_cuda else 'cpu')


    print(f"Main device (transformer/fc): {device}, GCN device (dgl): {gcn_device}")

    for num in range(1, 11):
        PATH_x_train = f'../../../data/data_splitClassifier/X_train{num}.csv'
        PATH_x_test  = f'../../../data/data_splitClassifier/X_test{num}.csv'
        PATH_x_val   = f'../../../data/data_splitClassifier/X_val{num}.csv'

        df_seq_train, y_train_tensor, y_true_train, list_num_train = func(PATH_x_train)
        df_seq_test,  y_test_tensor,  y_true_test,  list_num_test  = func(PATH_x_test)
        df_seq_val,   y_val_tensor,   y_true_val,   list_num_val   = func(PATH_x_val)

        node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')

        bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
        n_feats = atom_featurizer.feat_size('feat')
        print("atom feat dim:", n_feats)

        batch_size = 512

        def get_data(df):
            mols = [Chem.MolFromSmiles(x) for x in df['SMILES']]
            g = [mol_to_complete_graph(m, node_featurizer=node_featurizer) for m in mols]
            y = np.array(list((df['label'])), dtype=np.int64)
            return g, y

        model_trans = Transformer_test().to(device)
        gcn_net = GCNPredictor(in_feats=n_feats,
                               hidden_feats=[60, 20],
                               n_tasks=2,
                               predictor_hidden_feats=10,
                               predictor_dropout=0.5).to(gcn_device)
        transformer_dim = config.dim_model * config.pad_size
        gcn_out_dim = 40
        num_fc_out = 128
        model_tgcn = Model_TGCN(transformer_dim, gcn_out_dim, num_fc_out).to(device)

        def collate(sample):
            X_seq_list, list_num_list, graphs, labels, index = map(list, zip(*sample))
            X_seq = torch.stack(X_seq_list, dim=0)             # [B, 128] (int64)
            list_num_tensor = torch.stack(list_num_list, dim=0)  # [B, 103] (float32)
            batched_graph = dgl.batch(graphs)                  # DGL graph stays on CPU
            batched_graph.set_n_initializer(dgl.init.zero_initializer)
            batched_graph.set_e_initializer(dgl.init.zero_initializer)
            labels = torch.tensor(labels, dtype=torch.float32)  # [B]
            return X_seq, list_num_tensor, batched_graph, labels, index

        # DataLoader
        train_X = pd.read_csv(PATH_x_train)
        x_train, y_train = get_data(train_X)
        train_data = list(zip(df_seq_train, list_num_train, x_train, y_train, list(range(len(train_X)))))
        train_loader_ = DataLoader(train_data, batch_size=batch_size, shuffle=True,  collate_fn=collate, drop_last=False)

        test_X = pd.read_csv(PATH_x_test)
        x_test, y_test = get_data(test_X)
        test_data = list(zip(df_seq_test, list_num_test, x_test, y_test, list(range(len(test_X)))))
        test_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)

        val_X = pd.read_csv(PATH_x_val)
        x_val, y_val = get_data(val_X)
        val_data = list(zip(df_seq_val, list_num_val, x_val, y_val, list(range(len(val_X)))))
        val_loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)

        optimizer = torch.optim.Adam([
            {'params': gcn_net.parameters()},
            {'params': model_trans.parameters()},
            {'params': model_tgcn.parameters()}
        ], lr=0.001)
        bce = nn.BCELoss()

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = -1
        metrics_history = []
        save_dir_base = f'./model_origin/gcn_transformer_fc/{num}'
        os.makedirs(save_dir_base, exist_ok=True)
        best_ckpt_path = os.path.join(save_dir_base, 'best_checkpoint.pt')
        metrics_path = os.path.join(save_dir_base, 'metrics_history.json')

        for epoch in range(1, 500):
            gcn_net.train()
            model_trans.train()
            model_tgcn.train()
            train_loss_sum = 0.0
            train_correct = 0.0
            n_samples = 0

            for i, (X_seq, list_num, graph, labels, index) in enumerate(train_loader_):
                labels = labels.to(device)
                atom_feats = graph.ndata.pop('h').to(gcn_device)

                gcn_out_cpu = gcn_net(graph, atom_feats, model_use='a')

                gcn_out = gcn_out_cpu.to(device)

                X_seq = X_seq.to(device).float()
                list_num = list_num.to(device)

                y_t = model_trans(X_seq)  # [B, transformer_dim]

                y = model_tgcn(y_t, gcn_out, list_num)  # [B]
                y = y.view(-1)

                loss = bce(y, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.detach().item()
                preds = (y.detach() >= 0.5).float()
                train_correct += (preds == labels).float().sum().item()
                n_samples += labels.numel()

            train_loss_avg = train_loss_sum / (i + 1)
            train_acc = train_correct / n_samples if n_samples > 0 else 0.0

            def eval_loop(dataloader):
                gcn_net.eval()
                model_trans.eval()
                model_tgcn.eval()
                loss_sum = 0.0
                correct = 0.0
                total = 0
                plist = []
                with torch.no_grad():
                    for j, (X_seq, list_num, graph, labels, index) in enumerate(dataloader):
                        labels = labels.to(device)
                        atom_feats = graph.ndata.pop('h').to(gcn_device)
                        pred_cpu = gcn_net(graph, atom_feats, model_use='a')  # CPU
                        pred = pred_cpu.to(device)
                        X_seq = X_seq.to(device).float()
                        list_num = list_num.to(device)

                        y_t = model_trans(X_seq)
                        y = model_tgcn(y_t, pred, list_num).view(-1)
                        loss = bce(y, labels)
                        loss_sum += loss.item()
                        y_cpu = y.detach().cpu().numpy()
                        plist.extend(y_cpu)
                        preds = (y >= 0.5).float()
                        correct += (preds == labels).float().sum().item()
                        total += labels.numel()

                loss_avg = loss_sum / (j + 1)
                acc = correct / total if total > 0 else 0.0
                return acc, loss_avg, plist

            test_acc, test_loss, test_preds = eval_loop(test_loader_test)
            val_acc, val_loss, val_preds = eval_loop(val_loader_val)

            print(f"[split {num}] epoch {epoch:3d} | train loss {train_loss_avg:.4f} acc {train_acc:.4f} | "
                  f"test loss {test_loss:.4f} acc {test_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

            os.makedirs(f'./model_origin/gcn_transformer_fc/{num}/gcn/', exist_ok=True)
            os.makedirs(f'./model_origin/gcn_transformer_fc/{num}/transformer/', exist_ok=True)
            os.makedirs(f'./model_origin/gcn_transformer_fc/{num}/tgcn/', exist_ok=True)
            torch.save(gcn_net.state_dict(), f'./model_origin/gcn_transformer_fc/{num}/gcn/{epoch}_gcn.pt')
            torch.save(model_trans.state_dict(), f'./model_origin/gcn_transformer_fc/{num}/transformer/{epoch}_transformer.pt')
            torch.save(model_tgcn.state_dict(), f'./model_origin/gcn_transformer_fc/{num}/tgcn/{epoch}_tgcn.pt')

            metrics_history.append({
                'epoch': epoch,
                'train_loss': float(train_loss_avg),
                'train_acc': float(train_acc),
                'test_loss': float(test_loss),
                'test_acc': float(test_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
            })

            with open(metrics_path, 'w') as f:
                json.dump(metrics_history, f, indent=2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'gcn_state': gcn_net.state_dict(),
                    'trans_state': model_trans.state_dict(),
                    'tgcn_state': model_tgcn.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, best_ckpt_path)
                print(f"[split {num}] Saved NEW best checkpoint at epoch {epoch} (val_loss={val_loss:.4f}) -> {best_ckpt_path}")

            y_true_test = pd.read_csv(PATH_x_test, usecols=['label']).values
            t1, t2 = pd.DataFrame(test_preds, columns=['predict']), pd.DataFrame(y_true_test, columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data_origin/gcn_transformer_fc/{num}/test/', exist_ok=True)
            tt.to_csv(f'./pred_data_origin/gcn_transformer_fc/{num}/test/experiment_{epoch}_predicted_test_values.csv', index=False)

            y_true_val = pd.read_csv(PATH_x_val, usecols=['label']).values
            t1, t2 = pd.DataFrame(val_preds, columns=['predict']), pd.DataFrame(y_true_val, columns=['true'])
            tt = pd.concat([t1, t2], axis=1)
            os.makedirs(f'./pred_data_origin/gcn_transformer_fc/{num}/val/', exist_ok=True)
            tt.to_csv(f'./pred_data_origin/gcn_transformer_fc/{num}/val/experiment_{epoch}_predicted_valid_values.csv', index=False)

if __name__ == '__main__':
    main_()