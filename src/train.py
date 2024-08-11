import torch
from src.metrics import compute_average_accuracy, compute_average_auc, compute_average_f1_score

        
def train_model(model, loader, optimizer, criterion, cos_loss, epoch, device, logger, loss_lambda=[1,1]):
    # 모델을 학습하기 위한 함수
    cls_lambda = loss_lambda[0]
    cos_lambda = loss_lambda[1]
    model.train()
    train_loss = 0.0
    train_cls_loss, train_cos_loss = 0.0, 0.0
    
    total_pred = torch.empty((0, 100), device=device)
    total_true = torch.empty((0, 100), device=device)
    
    for batch_data in loader:
        optimizer.zero_grad()
        labels = batch_data['label'].float().squeeze().to(device)
        mask = batch_data['origin_mask'].reshape(-1,1).squeeze().to(device)
        output, next_visit_emb, final_visit_cls, final_visit = model(batch_data)
        y = torch.ones(output.size(0), dtype=torch.float, device=device)
        nz_cos_loss = cos_loss(output, next_visit_emb, y) * mask
        nz_num = mask.sum().item()
        cos_loss_mean = nz_cos_loss.sum() / nz_num
        cls_loss_mean = criterion(final_visit_cls.squeeze(), labels)
        loss = (cos_lambda * cos_loss_mean) + (cls_lambda * cls_loss_mean)
        
        y_pred = torch.sigmoid(final_visit_cls.squeeze())
        total_pred = torch.cat((total_pred, y_pred), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_cls_loss += cls_loss_mean.item()
        train_cos_loss += cos_loss_mean.item()
    
    avg_loss = train_loss / len(loader)
    avg_classi_loss = train_cls_loss / len(loader)
    avg_cos_loss = train_cos_loss / len(loader)

    acc = compute_average_accuracy(total_pred.cpu().detach(), 
                                   total_true.cpu().detach(), 
                                   reduction='mean')['accuracies']
    auc = compute_average_auc(total_pred.cpu().detach(), 
                              total_true.cpu().detach(), 
                              reduction='mean')
    f1_recall_prec = compute_average_f1_score(total_pred.cpu().detach(), 
                                  total_true.cpu().detach(), 
                                  reduction='macro')
    f1 = f1_recall_prec['average_f1_score']
    precision = f1_recall_prec['average_precision']
    recall = f1_recall_prec['average_recall']
    
    logger.info(f'[Epoch train {epoch}]: total loss: {avg_loss:.4f}, cls loss: {avg_classi_loss:.4f}, cos loss: {avg_cos_loss:.4f}')
    logger.info(f'[Epoch train {epoch}]: Acuuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    
    return {
            'loss': avg_loss, 
            'cls_loss': avg_classi_loss, 
            'cos_loss': avg_cos_loss
            }


@torch.no_grad()
def evaluate_model(model, loader, criterion, cos_loss, device, logger, epoch, mode='valid', loss_lambda=[1,1]):
    model.eval()
    cls_lambda = loss_lambda[0]
    cos_lambda = loss_lambda[1]
    val_loss = 0.0
    val_cls_loss, val_cos_loss = 0.0, 0.0
    
    total_pred = torch.empty((0, 100), device=device)
    total_true = torch.empty((0, 100), device=device)
    
    for batch_data in loader:
        labels = batch_data['label'].float().squeeze().to(device)
        mask = batch_data['origin_mask'].reshape(-1,1).squeeze().to(device)
        output, next_visit_emb, final_visit_cls, final_visit = model(batch_data)
        y = torch.ones(output.size(0), dtype=torch.float, device=device)
        nz_cos_loss = cos_loss(output, next_visit_emb, y) * mask
        nz_num = mask.sum().item()
        cos_loss_mean = nz_cos_loss.sum() / nz_num
        cls_loss_mean = criterion(final_visit_cls.squeeze(), labels)
        loss = (cos_lambda * cos_loss_mean) + (cls_lambda * cls_loss_mean)
        
        y_pred = torch.sigmoid(final_visit_cls.squeeze())
        total_pred = torch.cat((total_pred, y_pred), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)
                
        val_loss += loss.item()
        val_cls_loss += cls_loss_mean.item()
        val_cos_loss += cos_loss_mean.item()
    
    avg_val_loss = val_loss / len(loader)
    avg_classi_loss = val_cls_loss / len(loader)
    avg_cos_loss = val_cos_loss / len(loader)

    acc = compute_average_accuracy(total_pred.cpu().detach(), 
                                total_true.cpu().detach(), 
                                reduction='mean')['accuracies']
    auc = compute_average_auc(total_pred.cpu().detach(), 
                            total_true.cpu().detach(), 
                            reduction='mean')
    f1_recall_prec = compute_average_f1_score(total_pred.cpu().detach(), 
                                total_true.cpu().detach(), 
                                reduction='macro')
    f1 = f1_recall_prec['average_f1_score']
    precision = f1_recall_prec['average_precision']
    recall = f1_recall_prec['average_recall']
    logger.info(f'[Epoch {mode} {epoch}]: total loss: {avg_val_loss:.4f}, cls loss: {avg_classi_loss:.4f}, cos loss: {avg_cos_loss:.4f}')
    logger.info(f'[Epoch {mode} {epoch}]: Acuuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    return {
            'loss': avg_val_loss, 
            'cls_loss': avg_classi_loss, 
            'cos_loss': avg_cos_loss,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall
            }
    
    
   
def train_model2(model, loader, optimizer, criterion, epoch, device, logger):
    # 모델을 학습하기 위한 함수
    model.train()
    train_loss = 0.0
    
    total_pred = torch.empty((0, 100), device=device)
    total_true = torch.empty((0, 100), device=device)
    
    for batch_data in loader:
        optimizer.zero_grad()
        labels = batch_data['label'].float().squeeze().to(device)
        final_visit_cls, final_visit = model(batch_data)
        cls_loss_mean = criterion(final_visit_cls.squeeze(), labels)
        loss = cls_loss_mean
        
        y_pred = torch.sigmoid(final_visit_cls.squeeze())
        total_pred = torch.cat((total_pred, y_pred), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(loader)
    acc = compute_average_accuracy(total_pred.cpu().detach(), 
                                   total_true.cpu().detach(), 
                                   reduction='mean')['accuracies']
    auc = compute_average_auc(total_pred.cpu().detach(), 
                              total_true.cpu().detach(), 
                              reduction='mean')
    f1_recall_prec = compute_average_f1_score(total_pred.cpu().detach(), 
                                  total_true.cpu().detach(), 
                                  reduction='macro')
    f1 = f1_recall_prec['average_f1_score']
    precision = f1_recall_prec['average_precision']
    recall = f1_recall_prec['average_recall']
    
    logger.info(f'[Epoch train {epoch}]: total loss: {avg_loss:.4f}')
    logger.info(f'[Epoch train {epoch}]: Acuuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    
    return {
            'loss': avg_loss
            }


@torch.no_grad()
def evaluate_model2(model, loader, criterion, device, logger, epoch, mode='valid'):
    model.eval()
    val_loss = 0.0    
    total_pred = torch.empty((0, 100), device=device)
    total_true = torch.empty((0, 100), device=device)
    
    for batch_data in loader:
        labels = batch_data['label'].float().squeeze().to(device)
        final_visit_cls, final_visit = model(batch_data)
        cls_loss_mean = criterion(final_visit_cls.squeeze(), labels)
        loss = cls_loss_mean
        
        y_pred = torch.sigmoid(final_visit_cls.squeeze())
        total_pred = torch.cat((total_pred, y_pred), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)
        val_loss += loss.item()
    
    avg_val_loss = val_loss / len(loader)

    acc = compute_average_accuracy(total_pred.cpu().detach(), 
                                total_true.cpu().detach(), 
                                reduction='mean')['accuracies']
    auc = compute_average_auc(total_pred.cpu().detach(), 
                            total_true.cpu().detach(), 
                            reduction='mean')
    f1_recall_prec = compute_average_f1_score(total_pred.cpu().detach(), 
                                total_true.cpu().detach(), 
                                reduction='macro')
    f1 = f1_recall_prec['average_f1_score']
    precision = f1_recall_prec['average_precision']
    recall = f1_recall_prec['average_recall']
    logger.info(f'[Epoch {mode} {epoch}]: total loss: {avg_val_loss:.4f}')
    logger.info(f'[Epoch {mode} {epoch}]: Acuuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    return {
            'loss': avg_val_loss, 
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall
            }
    