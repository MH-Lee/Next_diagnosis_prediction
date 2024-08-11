import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        """
        Arguments:
        embeddings -- torch.Tensor of shape (2 * batch_size, embed_dim),
                      where 'embeddings' are concatenated outputs of [h_i and h_i+]
                      for all i in the batch, thus 2 * batch_size in total.

        Returns:
        loss -- scalar tensor, the computed SimCSE unsupervised contrastive loss
        """
        batch_size = embeddings.size(0) // 2

        # Cosine similarity matrix calculation
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2) / self.temperature

        # Extracting the diagonal elements, which are the similarities between each h_i and its corresponding h_i+
        positives = sim_matrix.diag()[1::2]  # start from index 1 with steps of 2

        # Summation of exponentials of the cosine similarities for each row, excluding self-similarity
        sum_exp = sim_matrix.exp().sum(dim=1) + sim_matrix.exp().sum(dim=0) - sim_matrix.exp().diag()

        # Negative log likelihood loss computation for each positive pair
        loss = -torch.log(positives / sum_exp[::2])  # take every second element starting from 0

        return loss.mean()  # Mean loss over all positive pairs in the batch
    

# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, embeddings_1, embeddings_2):
#         # 임베딩 정규화
#         embeddings_1 = F.normalize(embeddings_1, p=2, dim=1)
#         embeddings_2 = F.normalize(embeddings_2, p=2, dim=1)
#         # 코사인 유사도 계산
#         cosine_sim = torch.matmul(embeddings_1, embeddings_2.transpose(1,0)) / self.temperature
#         # 대각선(긍정적 쌍)에 대한 손실 계산
#         batch_size = embeddings_1.size(0)
#         labels = torch.arange(batch_size, device=embeddings_1.device)
#         loss = F.cross_entropy(cosine_sim, labels)
#         return loss
    

class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=128, device='cuda:0'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat.addmm_(x, self.centers.t(), beta = 1, alpha = -2,)

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


class BalancedBinaryCrossEntropyLoss(nn.Module):
    def __init__(
            self, alpha: float = None, device: str = "cpu"
    ):
        super(BalancedBinaryCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.device = device
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: Tensor, y: Tensor):
        bce_loss = self.bce_loss_fn(logits, y)
        weight = self.get_weight(y)

        return torch.mean(weight * bce_loss)

    def get_weight(self, y: Tensor):
        if self.alpha is None:
            return torch.ones_like(y)
        else:
            return self.alpha * y + (1 - self.alpha) * (1 - y)
        

class FocalLoss(nn.Module):
    # TODO: implement focal loss. if alpha is not None, use balanced focal loss.
    def __init__(self, gamma: float, alpha: float = None, device: str = "cpu"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

    def forward(self, logits: Tensor, y: Tensor):
        y_prob = torch.sigmoid(logits)
        fl = (
                - y * torch.pow(1 - y_prob, self.gamma) * torch.log(y_prob) -
                (1 - y) * torch.pow(y_prob, self.gamma) * torch.log(1 - y_prob)
        )
        if self.alpha is not None:
            weight = self.get_weight(y)
            fl = weight * fl

        return torch.mean(fl)

    def get_weight(self, y: Tensor):
        if self.alpha is None:
            return torch.ones_like(y)
        else:
            return self.alpha * y + (1 - self.alpha) * (1 - y)