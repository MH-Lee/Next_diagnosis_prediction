import torch
import torch.nn as nn
from src.attn import FixedPositionalEncoding, LearnablePositionalEncoding, TemporalEmbedding, Decoder


class CustomTransformerModel(nn.Module):
    def __init__(self, code_size, ninp, nhead, nlayers, dropout=0.5, device='cuda:0', pe='fixed'):
        super(CustomTransformerModel, self).__init__()
        self.ninp = ninp  # 축소된 차원 및 Transformer 입력 차원
        self.device = device
        # 차원 축소를 위한 선형 레이어
        self.pre_embedding = nn.Embedding(code_size + 1, self.ninp)
        self.pe = pe
        if self.pe == 'fixed':
            self.pos_encoder = FixedPositionalEncoding(ninp, dropout)
        elif self.pe == 'time_feature':
            self.pos_encoder = TemporalEmbedding(ninp, embed_type='embed', dropout=dropout)
        else:
            self.pos_encoder = LearnablePositionalEncoding(ninp, dropout)
            
        self.transformer_decoder = Decoder(ninp, nhead, nlayers, dropout)
        self.decoder = nn.Linear(ninp, ninp, bias=False)  # 최종 출력 차원을 설정 (여기서는 ninp로 설정)
        
        self.classification_layer = nn.Linear(ninp, 100)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.pre_embedding.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.xavier_uniform_(self.classification_layer.weight) 
        
    def forward(self, batch_data):
        # 차원 축소
        origin_visit = batch_data['origin_visit'].to(self.device)
        next_visit = batch_data['next_visit'].to(self.device)
        mask = (1- batch_data['origin_mask']).unsqueeze(2).to(self.device)
        mask_code = batch_data['origin_mask_code'].unsqueeze(3).to(self.device)
        next_mask_code = batch_data['next_mask_code'].unsqueeze(3).to(self.device)
        mask_final = batch_data['origin_mask_final'].unsqueeze(2).to(self.device)
        
        origin_visit_emb = (self.pre_embedding(origin_visit) * mask_code).sum(dim=2)
        next_visit_emb = (self.pre_embedding(next_visit) * next_mask_code).sum(dim=2)
        # 위치 인코딩 및 Transformer 디코더 적용
        if self.pe == 'time_feature':
            time_feature = batch_data['time_feature'].to(self.device)
            src, temporal_embed = self.pos_encoder(origin_visit_emb, time_feature)
        else:
            src = self.pos_encoder(origin_visit_emb)
        seq_len = src.shape[1]
        src_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=self.device), diagonal=1)
        attn_output = self.transformer_decoder(src, attention_mask=src_mask)
        output = self.decoder(attn_output)
        output_batch, output_visit_num, output_dim = output.size()
        next_visit_output_batch, next_visit_output_num, next_visit_output_dim = next_visit_emb.size()
        
        temp_output = output.reshape(output_batch, -1)
        temp_next_visit = next_visit_emb.reshape(next_visit_output_batch, -1)
        final_visit = (output * mask_final).mean(dim=1)
        classification_result = self.classification_layer(final_visit)
        return temp_output, temp_next_visit, classification_result, final_visit
    


class CustomTransformerModel2(nn.Module):
    def __init__(self, code_size, ninp, nhead, nlayers, dropout=0.5, device='cuda:0', pe='fixed'):
        super(CustomTransformerModel2, self).__init__()
        self.ninp = ninp  # 축소된 차원 및 Transformer 입력 차원
        self.device = device
        # 차원 축소를 위한 선형 레이어
        self.pre_embedding = nn.Embedding(code_size + 1, self.ninp)
        self.pe = pe
        if self.pe == 'fixed':
            self.pos_encoder = FixedPositionalEncoding(ninp, dropout)
        elif self.pe == 'time_feature':
            self.pos_encoder = TemporalEmbedding(ninp, embed_type='embed', dropout=dropout)
        else:
            self.pos_encoder = LearnablePositionalEncoding(ninp, dropout)
            
        self.transformer_decoder = Decoder(ninp, nhead, nlayers, dropout)
        self.decoder = nn.Linear(ninp, ninp, bias=False)  # 최종 출력 차원을 설정 (여기서는 ninp로 설정)
        
        self.classification_layer = nn.Linear(ninp, 100)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.pre_embedding.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.xavier_uniform_(self.classification_layer.weight) 
        
    def forward(self, batch_data):
        # 차원 축소
        origin_visit = batch_data['origin_visit'].to(self.device)
        mask = (1- batch_data['origin_mask']).unsqueeze(2).to(self.device)
        mask_code = batch_data['origin_mask_code'].unsqueeze(3).to(self.device)
        mask_final = batch_data['origin_mask_final'].unsqueeze(2).to(self.device)
        
        origin_visit_emb = (self.pre_embedding(origin_visit) * mask_code).sum(dim=2)
        # 위치 인코딩 및 Transformer 디코더 적용
        if self.pe == 'time_feature':
            time_feature = batch_data['time_feature'].to(self.device)
            src, temporal_embed = self.pos_encoder(origin_visit_emb, time_feature)
        else:
            src = self.pos_encoder(origin_visit_emb)
        seq_len = src.shape[1]
        src_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=self.device), diagonal=1)
        attn_output = self.transformer_decoder(src, attention_mask=src_mask)
        output = self.decoder(attn_output)
        final_visit = (output * mask_final).mean(dim=1)
        classification_result = self.classification_layer(final_visit)
        return classification_result, final_visit