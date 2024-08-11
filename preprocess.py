import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import argparse

def pad_sequence(seq_diagnosis_codes, maxlen, maxcode):
    lengths = len(seq_diagnosis_codes)
    diagnosis_codes = np.zeros((maxlen, maxcode), dtype=np.int64)
    seq_mask_code = np.zeros((maxlen, maxcode), dtype=np.int8)
    seq_mask = np.zeros((maxlen), dtype=np.int8)
    seq_mask_final = np.zeros((maxlen), dtype=np.int8)
    for pid, subseq in enumerate(seq_diagnosis_codes):
        for tid, code in enumerate(subseq):
            diagnosis_codes[pid, tid] = code
            seq_mask_code[pid, tid] = 1
    seq_mask[:lengths] = 1
    seq_mask_final[lengths - 1] = 1
    return diagnosis_codes, seq_mask_code, seq_mask, seq_mask_final


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

def preprocess_ehr(data_dict_d, max_visits_length, max_code_len):
    new_data_dict_d = {}
    for sample_id, data in tqdm(data_dict_d.items()):
        data_dict_new = {}
        # pad_seq, seq_mask_code = pad_sequence(data['year'], max_visits_length, max_code_len)
        pad_seq, seq_mask_code, seq_mask, seq_mask_final = pad_sequence(data['code_seq'], max_visits_length, max_code_len)
        data_dict_new['code_index'] = pad_seq
        data_dict_new['code'] = data['code_seq']
        data_dict_new['time'] = data['datetime']
        data_dict_new['timedelta'] = data['timedelta']
        time_feature = np.array([[timestamp.year, timestamp.month, timestamp.day, timestamp.date().isocalendar()[1]] for timestamp in data['datetime']])
        data_dict_new['time_feature'] = np.pad(time_feature, pad_width=((0, max_visits_length - time_feature.shape[0]),(0,0)))
        data_dict_new['year_month'] =  data['ym']
        data_dict_new['seq_mask'] = seq_mask
        data_dict_new['seq_mask_final'] = seq_mask_final
        data_dict_new['seq_mask_code'] = seq_mask_code
        unique_year_month = np.unique(data_dict_new['year_month'])
        if sklearn.__version__ == '1.5.0':
            encoder = OneHotEncoder(categories=[unique_year_month], sparse=False, handle_unknown='ignore')
        else:
            encoder = OneHotEncoder(categories=[unique_year_month], sparse_output=False, handle_unknown='ignore')

        year_month_onehot =  encoder.fit_transform(np.array(data_dict_new['year_month']).reshape(-1,1))
        last_year_visit = keep_last_one_in_columns(year_month_onehot)
        data_dict_new['year_month_onehot'] = np.pad(year_month_onehot, pad_width=((0, max_visits_length - year_month_onehot.shape[0]), (0,0)))
        data_dict_new['last_year_month_onehot'] = np.pad(last_year_visit, pad_width=((0, max_visits_length - year_month_onehot.shape[0]), (0,0)))
        data_dict_new['label'] = data['top100_label_bin']       
        new_data_dict_d[sample_id] = data_dict_new
    return new_data_dict_d
    

if __name__ == '__main__':
    # Load data
    parser = argparse.ArgumentParser(description='preprocess for mimiciv')
    parser.add_argument('--path', type=str, default='./data/', help='path to data')
    parser.add_argument('--save_path', type=str, default='./data/', help='path to save')

    args = parser.parse_args()
    
    with open(os.path.join(args.path, 'mimiciv_code2idx_nd.pkl'), 'rb') as f:
        dtype_dict = pickle.load(f)
    f.close()

    with open(os.path.join(args.path, 'filtered_data_seq_visit_o3.pkl'), 'rb') as f:
        data_dict_d = pickle.load(f)
    f.close()
    
    data_dict_d2 = dict()
    for sample_id, visits in tqdm(data_dict_d.items()):
        if (len(visits['seq']) <= 50) and (len(visits['top100_label']) != 0):
            data_dict_d2[sample_id] = visits
        else:
            continue
        
    total_labels = []
    length_list = []
    code_length_list = []
    for sample_id, visits in tqdm(data_dict_d2.items()):
        # 레이블 추가
        total_label = visits['top100_label_bin'].tolist()[0]
        total_labels.append(total_label)
        length_list.append(len(visits['code_seq']))
        code_length_list.append(max([len(seq)for seq in visits['code_seq']]))
        
    max_visits_length = max(length_list)
    max_index = max(dtype_dict.values())
    max_code_len = max(code_length_list)
    print('max_index:', max_index+1)
    print('max_visit:', max_visits_length)
    print('max_code_len:', max_code_len)
    
    date_str = datetime.now().strftime("%Y%m%d")
    new_data_dict_d = preprocess_ehr(data_dict_d2, max_visits_length, max_code_len)
    with open(os.path.join(args.save_path, f'preprocessed_nd_{date_str}.pkl'), 'wb') as f:
        pickle.dump(new_data_dict_d, f)
    f.close()
        
    