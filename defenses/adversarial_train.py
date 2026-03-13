import torch
import torch.nn as nn
import wandb

from classifiers.get_model import get_model
from classifiers.abstract_classifier import AbstractClassifier
from utils import wandb_helpers

def apply_adversarial_training_defense(config, model:AbstractClassifier):
    
    class AdversarialTraining(model.__class__): 
        
        def __init__(self, config):
            super(AdversarialTraining, self).__init__(config)
            
            self.method = self.config.defense.method
        
            self.targeted = self.config.defense.targeted
            
            target_model = self # The target of the pgd attack is the model itself.
                
            ### Setup the attack
            if self.method == "pgd":
                self.attack = PGDAttack(
                    model=target_model,
                    loss_fn=nn.CrossEntropyLoss(),
                    epsilon=config.defense.pgd.epsilon,
                    step_size=config.defense.pgd.step_size,
                    iterations=config.defense.pgd.iterations,
                    targeted=self.targeted,
                    random_start=config.defense.pgd.random_start,
                    min_clamp=-1,
                    max_clamp=1
                )
            elif self.method == "fgsm":
                self.attack = FGSM(
                    model=target_model,
                    loss_fn=nn.CrossEntropyLoss(),
                    epsilon=config.defense.fgsm.epsilon,
                    targeted=self.targeted,
                    min_clamp=-1,
                    max_clamp=1
                )
            else:
                raise ValueError(f"Unknown attack method: {self.method}")

        def train_one_epoch(self, train_loader):
            self.train()
            total_loss = 0
            loss_calculated = 0
            
            attack_success_sum = 0
            total_loss_normal = 0
            total_loss_adversarial = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                if self.config.defense.loss_weighting.standard:
                    output = self(data)
                    loss_normal = self.get_loss(output, target)
                else:
                    loss_normal = 0
                
                if self.targeted:
                    # choose another target randomly
                    attack_target = torch.randint(0, self.config.dataset.n_classes, (target.shape[0],)).to(self.device)
                    
                    # if the target is the same as the original label, choose another one
                    same_target = (attack_target == target)
                    attack_target[same_target] = (attack_target[same_target] + 1) % self.config.dataset.n_classes
                else:
                    attack_target = target    
                
                data_adv = self.attack(data, attack_target)
                
                output_adv = self(data_adv)
                
                pred_adv = torch.argmax(output_adv, dim=1)
                
                if self.targeted:
                    # For targeted attacks, we want to see if the model is fooled into predicting the target class
                    attack_success_sum += (pred_adv == attack_target).sum()
                else:
                    # For untargeted attacks, we want to see if the model is fooled into predicting a different class
                    attack_success_sum += (pred_adv != target).sum()
                    
                if self.targeted:
                    loss_adv = self.get_loss(output_adv, attack_target)
                else:
                    loss_adv = self.get_loss(output_adv, target)
                
                loss = self.config.defense.loss_weighting.standard * loss_normal + self.config.defense.loss_weighting.adversarial * loss_adv

                if loss_normal != 0:
                    total_loss_normal += loss_normal.item()
                    
                total_loss_adversarial += loss_adv.item()
                total_loss += loss.item()          

                loss_calculated += len(data)

                if self.config.training.verbose == 2:
                    print("loss: ", loss.item())

                loss.backward()
                self.optimizer.step()
                self.train_step += 1

            train_loss = total_loss / loss_calculated
            
            train_loss_normal = total_loss_normal / loss_calculated
            train_loss_adversarial = total_loss_adversarial / loss_calculated
            
            attack_success_percentage = attack_success_sum / loss_calculated
            
            print(f"(normal: {train_loss_normal:.4f}, adversarial: {train_loss_adversarial:.4f}), attack_success_percentage: {attack_success_percentage:.4f}")
            
            if wandb.run is not None:
                wandb.log({"train_step": self.train_step,
                           "train_loss_adversarial": train_loss_adversarial, 
                           "train_loss_normal": train_loss_normal, 
                           "attack_success_percentage": attack_success_percentage})
            
            return train_loss

    advesarial_defended_model = AdversarialTraining(config)
    
    return advesarial_defended_model

class PGDAttack:
    def __init__(self, model, loss_fn, epsilon:float, step_size:float, iterations:int, targeted:bool, random_start:bool, min_clamp=-1, max_clamp=1):
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.step_size = step_size
        self.iterations = iterations
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.targeted = targeted
        self.device = self.model.device
        self.random_start = random_start
        
        self.model.zero_grad()
        
    # Modified from torchattacks: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py
    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.epsilon, self.epsilon
            )
            adv_images = torch.clamp(adv_images, min=self.min_clamp, max=self.max_clamp).detach()

        for _ in range(self.iterations):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -self.loss_fn(outputs, labels)
            else:
                cost = self.loss_fn(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.step_size * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + delta, min= self.min_clamp, max=self.max_clamp).detach()

        return adv_images

# Inspired by: https://github.com/Harry24k/FGSM-pytorch/blob/master/FGSM.ipynb

class FGSM:
    def __init__(self, model, loss_fn, epsilon:float, targeted:bool, min_clamp=-1, max_clamp=1):
        self.model = model
        self.device = model.device
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.targeted = targeted
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp

    def __call__(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        images.requires_grad = True
                
        outputs = self.model(images)
        
        self.model.zero_grad()
        
        if self.targeted:
            cost = -self.loss_fn(outputs, labels).to(self.model.device)
        else:
            cost = self.loss_fn(outputs, labels).to(self.model.device)
        
        cost.backward()
        
        attack_images = images + self.epsilon * images.grad.sign()
        attack_images = torch.clamp(attack_images, self.min_clamp, self.max_clamp)
        
        attack_images = attack_images.detach()
        attack_images.requires_grad = False
        
        return attack_images