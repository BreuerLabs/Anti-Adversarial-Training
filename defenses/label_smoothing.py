from classifiers.abstract_classifier import AbstractClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

def apply_label_smoothing_defense(config, model:AbstractClassifier):
    
    class LabelSmoothing(model.__class__):
        
        def __init__(self, config):
            super(LabelSmoothing, self).__init__(config)
            if config.model.criterion == "crossentropy":
                self.criterion = CrossEntropyLoss(label_smoothing=config.defense.alpha)
                self.criterionSum = CrossEntropyLoss(label_smoothing=config.defense.alpha, reduction='sum')

            if config.defense.ls_scheduler: # Start label smoothing slowly, as in Struppek et al. 2024
                def ls_scheduler(epoch):
                    if epoch < 50:
                        return 0.0
                    elif epoch < 75:
                        factor = 1 / (config.model.hyper.epochs - 50)
                        return config.defense.alpha * factor * (epoch + 1 - 50)
                    else:
                        return config.defense.alpha
                self.ls_scheduler = ls_scheduler
            else:
                self.ls_scheduler = None
            self.epoch = 0

        def train_one_epoch(self, train_loader):
            if self.ls_scheduler: # update label smoothing value before each epoch
                self.epoch += 1
                self.criterion.label_smoothing = self.ls_scheduler(self.epoch)
                self.criterionSum.label_smoothing = self.ls_scheduler(self.epoch)
                if self.config.training.wandb.track:
                    wandb.log({"label_smoothing": self.criterion.label_smoothing})

            train_loss = super(LabelSmoothing, self).train_one_epoch(train_loader)
            return train_loss

    label_smoothing_defended_model = LabelSmoothing(config)
    return label_smoothing_defended_model


class CrossEntropyLoss(nn.Module):

    def __init__(self, label_smoothing=0.0, reduction = "mean"):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, labels):
        confidence = 1.0 - self.label_smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        
        if self.reduction == "mean":
            result = torch.mean(loss) / num_batch
        
        elif self.reduction == "sum":
            result = torch.sum(loss) / num_batch
        
        else:
            raise ValueError("No valid reduction")
        
        return result
