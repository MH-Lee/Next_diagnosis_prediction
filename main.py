import numpy as np
import os 
import pickle
import sys
import random
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime
import os.path as osp
import time
from sklearn.model_selection import train_test_split
from src.utils import get_logger, CustomDataset
from src.model import CustomTransformerModel, CustomTransformerModel2
from src.loss import BalancedBinaryCrossEntropyLoss, FocalLoss, DistanceCorrelation
from src.train import train_model, evaluate_model, train_model2, evaluate_model2
from torch.utils.data import DataLoader
from pprint import pprint
import warnings

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore")


class Runner:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger(args.name, args.log_dir, args.config_dir)
        self.logger.info(vars(args))
        pprint(vars(args))
        if self.args.device != 'cpu' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
        self.seed = args.seed
        self.logger.info(f'device: {self.device}')
        self.dtype_dict = pickle.load(open(osp.join(args.data_dir, 'mimiciv_code2idx_nd.pkl'), 'rb'))
        self.data_dict_d = pickle.load(open(osp.join(args.data_dir, 'preprocessed_nd_20240702.pkl'), 'rb'))
        self.loss_lambda = args.loss_lambda
        print(len(list(self.data_dict_d.keys())))
        self.indices_dir = osp.join(self.args.data_dir, 'indices')
        self.date_str = datetime.now().strftime("%Y%m%d")
        self.make_indices()
        self.load_data()
        if self.args.model_name == 'ehr_gpt':
            self.model = CustomTransformerModel(code_size=len(self.dtype_dict), ninp=args.input_dim,
                                                nhead=args.n_heads, nlayers=args.n_layers, dropout=args.dropout, 
                                                device=self.device, pe=args.pe_type).to(self.device)
            self.cos_emb_loss = nn.CosineEmbeddingLoss(reduction='none')
            # self.cos_emb_loss = DistanceCorrelation(device=self.device)
        elif self.args.model_name == 'ehr_gpt_v2':
            self.model = CustomTransformerModel2(code_size=len(self.dtype_dict), ninp=args.input_dim,
                                                 nhead=args.n_heads, nlayers=args.n_layers, dropout=args.dropout, 
                                                 device=self.device, pe=args.pe_type).to(self.device)
        else:
            raise NotImplementedError(f'{self.args.name} is not implemented')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        filename_format = f'{self.args.name}_inputdim:{self.args.input_dim}_nlayer:{self.args.n_layers}_nheads:{self.args.n_heads}_seed:{self.seed}_{self.args.pe_type}_{self.args.loss_type}'
        if self.args.loss_type == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
            self.filename_format = filename_format
        elif self.args.loss_type == 'balanced_bce':
            self.loss = BalancedBinaryCrossEntropyLoss(alpha=args.alpha, device=self.device)
            self.filename_format = filename_format + f'_alpha:{args.alpha}'
        elif self.args.loss_type == 'focalloss':
            self.loss = FocalLoss(gamma=args.gamma, alpha=args.alpha, device=self.device)
            self.filename_format = filename_format + f'_alpha:{args.alpha}_gamma:{args.gamma}'
        else:
            raise NotImplementedError(f'{self.args.loss_type} is not implemented')

        
    def make_indices(self):
        if osp.isfile(osp.join(self.indices_dir, f'train_indices_{self.seed}.pkl')):
            with open(osp.join(self.indices_dir, f'train_indices_{self.seed}.pkl'), 'rb') as f:
                self.train_indices = pickle.load(f)
            f.close()
            with open(osp.join(self.indices_dir, f'valid_indices_{self.seed}.pkl'), 'rb') as f:
                self.valid_indices = pickle.load(f)
            f.close()
            with open(osp.join(self.indices_dir, f'test_indices_{self.seed}.pkl'), 'rb') as f:
                self.test_indices = pickle.load(f)
            f.close()
        else:
            train_indices, test_indices = train_test_split(list(self.data_dict_d.keys()), 
                                                           test_size=0.1, 
                                                           random_state=self.seed)
            train_indices, valid_indices = train_test_split(train_indices, 
                                                            test_size=(len(test_indices)/len(train_indices)), 
                                                            random_state=self.seed)
            with open(osp.join(self.indices_dir, f'train_indices_{self.seed}.pkl'), 'wb') as f:
                pickle.dump(train_indices, f)
            f.close()

            with open(osp.join(self.indices_dir, f'valid_indices_{self.seed}.pkl'), 'wb') as f:
                pickle.dump(valid_indices, f)
            f.close()

            with open(osp.join(self.indices_dir, f'test_indices_{self.seed}.pkl'), 'wb') as f:
                pickle.dump(test_indices, f)
            f.close()
            
            self.train_indices, self.valid_indices, self.test_indices = train_indices, valid_indices, test_indices

    def load_data(self):
        train_data = {}
        valid_data = {}
        test_data = {}
        for sample in tqdm(self.train_indices):
            train_data[sample] = self.data_dict_d[sample]

        for sample in tqdm(self.valid_indices):
            valid_data[sample] = self.data_dict_d[sample]

        for sample in tqdm(self.test_indices):
            test_data[sample] = self.data_dict_d[sample]
        
        train_dataset = CustomDataset(train_data)
        valid_dataset = CustomDataset(valid_data)
        test_dataset = CustomDataset(test_data)
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        
    def fit(self):
        tr_loss_list, tr_cls_list, tr_cos_list = list(), list(), list()
        counter = 0
        best_score = 0.0
        best_epoch = 0
        model_save_path = ''
        
        for epoch in tqdm(range(self.args.max_epoch)):
            if self.args.model_name == 'ehr_gpt':
                train_log = train_model(model=self.model, loader=self.train_loader, optimizer=self.optimizer,
                                        criterion=self.loss, cos_loss=self.cos_emb_loss, epoch=epoch, 
                                        device=self.device, logger=self.logger, loss_lambda=self.loss_lambda)
                valid_log = evaluate_model(model=self.model, loader=self.valid_loader, criterion=self.loss, 
                                        cos_loss=self.cos_emb_loss, device=self.device, logger=self.logger, 
                                        epoch=epoch, mode='valid', loss_lambda=self.loss_lambda)
                tr_loss_list.append(train_log['loss']); tr_cls_list.append(train_log['cls_loss']); tr_cos_list.append(train_log['cos_loss'])
            elif self.args.model_name == 'ehr_gpt_v2':
                train_log = train_model2(model=self.model, loader=self.train_loader, optimizer=self.optimizer,
                                         criterion=self.loss, epoch=epoch, device=self.device, logger=self.logger)
                valid_log = evaluate_model2(model=self.model, loader=self.valid_loader, criterion=self.loss, 
                                            device=self.device, logger=self.logger, epoch=epoch, mode='valid')
                tr_loss_list.append(train_log['loss'])
                
            current_score = valid_log['auc']
            if current_score > best_score:
                if osp.isfile(model_save_path):
                    os.remove(model_save_path)
                best_epoch = epoch
                counter = 0
                best_score = valid_log['auc']
                model_filename = self.filename_format  + f'_best_epoch:{best_epoch}.pt'
                model_save_path = os.path.join(self.args.checkpoint_dir, self.date_str, model_filename)
                torch.save(self.model.state_dict(), f'{model_save_path}')
            else:
                counter += 1
                self.logger.info(f"Early stopping counter: {counter}/{self.args.patience}")
              
            if counter >= self.args.patience:
                self.logger.info(f"Early stopping triggered at epoch {best_epoch}")
                self.logger.info(f"Best Combined Score (AUC): {best_score:.4f}")
                break
        
        if self.args.model_name == 'ehr_gpt':
            test_log = evaluate_model(model=self.model, loader=self.test_loader, criterion=self.loss, 
                                        cos_loss=self.cos_emb_loss, device=self.device, logger=self.logger, 
                                        epoch=epoch, mode='test', loss_lambda=self.loss_lambda)
        elif self.args.model_name == 'ehr_gpt_v2':
            test_log = evaluate_model2(model=self.model, loader=self.test_loader, criterion=self.loss, 
                                        device=self.device, logger=self.logger, epoch=epoch, mode='test')
            
        with open(os.path.join(self.args.log_dir, model_filename + f'_best_epoch:{best_epoch}.txt'), 'w') as f:
            f.write('\n')
            f.write(str(test_log))
            f.write('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EHR mimic-iv train model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')
    parser.add_argument('--config_dir', type=str, default='./config/', help='config directory')
    parser.add_argument('--name', type=str, default='ehr_gpt', help='model name')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/', help='model directory')
    parser.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--loss_lambda', type=list, default=[1,1], help='loss lambda')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=777, help='seed')
    parser.add_argument('--input_dim', type=int, default=128, help='input dimension')
    parser.add_argument('--n_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--loss_type', type=str, default='bce', help='loss type')
    parser.add_argument('--pe_type', type=str, default='time_feature', help='position encoding type')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for balanced bce or focal loss')
    parser.add_argument('--gamma', type=float, default=2, help='gamma for focal loss')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping')
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    date_dir = datetime.today().strftime("%Y%m%d")
    
    args.model_name = args.name
    args.name = args.name + '_' +  date_dir + '_' + time.strftime('%H:%M:%S')
    args.log_dir = osp.join(args.log_dir, date_dir)
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(osp.join(args.checkpoint_dir, date_dir), exist_ok=True)
    os.makedirs(osp.join(args.data_dir, 'indices'), exist_ok=True)
    
    model = Runner(args)
    model.fit()