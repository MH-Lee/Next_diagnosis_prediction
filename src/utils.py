import torch
import sys
import json
import logging, logging.config
import numpy as np
from torch.utils.data import Dataset

def pad_sequence(seq_diagnosis_codes, maxlen, maxcode):
    diagnosis_codes = np.zeros((maxlen, maxcode), dtype=np.int64)
    seq_mask_code = np.zeros((maxlen, maxcode), dtype=np.float32)
    for pid, subseq in enumerate(seq_diagnosis_codes):
        for tid, code in enumerate(subseq):
            diagnosis_codes[pid, tid] = code
            seq_mask_code[pid, tid] = 1
    return diagnosis_codes, seq_mask_code

def keep_last_one_in_columns(a):
    # 결과 배열 초기화
    result = np.zeros_like(a)
    # 각 열에 대해 반복
    for col_index in range(a.shape[1]):
        # 현재 열 추출
        column = a[:, col_index]
        # 이 열에서 마지막 '1' 찾기
        last_one_idx = np.max(np.where(column == 1)[0]) if 1 in column else None
        if last_one_idx is not None:
            result[last_one_idx, col_index] = 1
    return result

def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open(config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)
	return logger


class CustomDataset(Dataset):
    def __init__(self, data):       
        self.keys = list(data.keys())  # 딕셔너리의 키 목록 저장
        self.data = data  # 딕셔너리에서 데이터만 추출하여 저장

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x는 현재 방문까지의 모든 방문 데이터, y는 다음 방문 데이터
        padding_temp = torch.zeros((1, len(self.data[self.keys[idx]]['code_index'][0])), dtype=torch.long)
        
        origin_visit = torch.tensor(self.data[self.keys[idx]]['code_index'], dtype=torch.long)
        origin_mask = torch.tensor(self.data[self.keys[idx]]['seq_mask'], dtype=torch.long)
        origin_mask_final = torch.tensor(self.data[self.keys[idx]]['seq_mask_final'], dtype=torch.long)
        origin_mask_code = torch.tensor(self.data[self.keys[idx]]['seq_mask_code'], dtype=torch.long)
        
        next_visit = torch.cat((origin_visit[1:], padding_temp), dim=0)
        next_mask =torch.cat((origin_mask[1:], torch.tensor([0], dtype=torch.long)), dim=0)
        next_mask_code = torch.cat((origin_mask_code[1:], padding_temp), dim=0)
        
        visit_index = torch.tensor(self.data[self.keys[idx]]['year_month_onehot'], dtype=torch.long)
        last_visit_index = torch.tensor(self.data[self.keys[idx]]['last_year_month_onehot'], dtype=torch.float)
        time_feature = torch.tensor(self.data[self.keys[idx]]['time_feature'], dtype=torch.long)
        label_per_sample = self.data[self.keys[idx]]['label']
        key_per_sample = self.keys[idx]  # 해당 샘플의 키
        # 키 값도 함께 반환
        return {'sample_id': key_per_sample, 'origin_visit': origin_visit, 'next_visit': next_visit,\
                'origin_mask': origin_mask, 'origin_mask_final': origin_mask_final, 'next_mask': next_mask, \
                'origin_mask_code': origin_mask_code, 'next_mask_code': next_mask_code, 'time_feature': time_feature,\
                'label': label_per_sample}
        



