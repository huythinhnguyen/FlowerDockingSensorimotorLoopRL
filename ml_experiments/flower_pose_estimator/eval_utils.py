import numpy as np

from matplotlib import pyplot as plt

from Sensors.FlowerEchoSimulator.Spatializer import wrapToPi


import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def torchWrapToPi(x):
    return torch.fmod(x + torch.tensor(np.pi).to(DEVICE), torch.mul(torch.tensor(np.pi).to(DEVICE),2)) - torch.tensor(np.pi).to(DEVICE)

class WrapToPiMSELoss(nn.Module):
    def __init__(self):
        super(WrapToPiMSELoss, self).__init__()
        self.name = 'modpi_MSE'
        
    def forward(self, prediction, target):
        return torch.mean((torchWrapToPi(prediction - target))**2)
    
class WrapToPiMAELoss(nn.Module):
    def __init__(self):
        super(WrapToPiMAELoss, self).__init__()
        self.name = 'modpi_MAE'
        
    def forward(self, prediction, target):
        return torch.mean(torch.abs(torchWrapToPi(prediction - target)))
    
class WrapToPiMAPELoss(nn.Module):
    def __init__(self):
        super(WrapToPiMAPELoss, self).__init__()
        self.name = 'modpi_MAPE'
        
    def forward(self, prediction, target):
        return torch.mean(torch.abs(torchWrapToPi((prediction - target)/target)))

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.name = 'MSE'
        
    def forward(self, prediction, target):
        return torch.mean((prediction - target)**2)

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.name = 'MAE'
        
    def forward(self, prediction, target):
        return torch.mean(torch.abs(prediction - target))
    
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()
        self.name = 'MAPE'
        
    def forward(self, prediction, target):
        return torch.mean(torch.abs((prediction - target)/target))
    
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.name = 'RMSE'
        
    def forward(self, prediction, target):
        return torch.sqrt(torch.mean((prediction - target)**2))
    
class WrapToPiRMSELoss(nn.Module):
    def __init__(self):
        super(WrapToPiRMSELoss, self).__init__()
        self.name = 'modpi_RMSE'

    def forward(self, prediction, target):
        return torch.sqrt(torch.mean((torchWrapToPi(prediction - target))**2))
    

### Not Using Any of this ###
# I put everything within a training loop However.
# Also, Loss Balancing was not neccessary since all 3 tasks were learned sufficiently well with equal weights

### Not Using ###
class MultiTaskLoss(nn.Module):
    def __init__(self, loss_functions: list, init_loss_weights: list=None, loss_requires_grad: bool=False):
        super(MultiTaskLoss, self).__init__()
        self.loss_functions = loss_functions
        self.loss_names = []
        for loss in loss_functions:
            self.loss_names.append(loss.name)
        self.task_num = len(loss_functions)
        if not init_loss_weights: self.loss_weights = torch.nn.Parameter(torch.ones(self.task_num, requires_grad=loss_requires_grad))
        else: self.loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights, requires_grad=True))
        self.losses = []

    def forward(self, prediction, target):
        for i in range(self.task_num):
            self.losses.append(self.loss_weights[i] * self.loss_functions[i](prediction[:,i], target[:,i]))
        return torch.sum(torch.stack(self.losses))

### Not Using ###
class MultiTaskLossSoftAdapt(MultiTaskLoss):
    def __init__(self, *args, **kwargs):
        super(MultiTaskLossSoftAdapt, self).__init__(*args, **kwargs)
        self.name = 'MultiTaskLossSoftAdapt'
        self.previous_losses = None
        self.softmax = nn.Softmax(dim=-1)
        self.loss_weights = self.softmax(self.loss_weights)
        self.losses = []

    # def update_weights(self, current_losses):
    #     # if previous losses are None, set them to current losses, not update weights
    #     if self.previous_losses is None:
    #         self.previous_losses = current_losses
    #         return None
        
    #     # weight is updated with
    #     # wj = exp(Lj/(Lj_prev+1e-4) - mu) / sum(exp(Lk/(Lk_prev+1e-4) - mu))
        
    #     # mu = max(L/(L_prev))
    #     mu = torch.max(current_losses / (self.previous_losses))
    #     demnominator = torch.sum(torch.exp( torch.div(current_losses, (self.previous_losses + 1e-4)) - mu))
    #     ### Main update loop
    #     for i, (loss, prev_loss) in enumerate(zip(current_losses, self.previous_losses)):
    #         self.loss_weights[i] = torch.div(torch.exp(torch.div(loss, (prev_loss + 1e-4)) - mu) , demnominator)

    #     # update previous losses
    #     self.previous_losses = current_losses
    #     return None

    def update_weights(self, current_losses):
        if self.previous_losses is None:
            self.previous_losses = current_losses
            return None
        
        # weight is updated with
        # wj = exp(Lj/(Lj_prev+1e-4) - mu) / sum(exp(Lk/(Lk_prev+1e-4) - mu))
        # mu = max(L/(L_prev))
        mu = torch.max(current_losses / (self.previous_losses))
        self.loss_weights = self.softmax(torch.div(current_losses, (self.previous_losses + 1e-8)) - mu)
        self.previous_losses = current_losses
        return None
    
    def forward(self, prediction, target):
        self.losses = []
        for i in range(self.task_num):
            self.losses.append(self.loss_weights[i] * self.loss_functions[i](prediction[:,i], target[:,i]))
        self.update_weights(torch.stack(self.losses))
        return torch.sum(torch.stack(self.losses))


### Not Using ###
def train_with_gradnorm(model, X_train, y_train, X_val, y_val, loss_fn, epochs, batch_size, learning_rate, weight_decay, device, gradnorm_layer, optimizer_name=None, fixed_momentum=0.5,
                        gradnorm_learning_rate = 1e-5, gradnorm_alpha=1.5, logging_weights = True):
    model = model.to(device)
    criterion = loss_fn # hopefully the based function!
    if optimizer_name==None: optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name=='SGD': optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.5)


    train_losses = []
    val_losses = []
    iter = 0
    caches = {'weights': [], 'loss_ratio': [], 'losses': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for i in range(0, X_train.shape[0], batch_size):
            inputs = torch.from_numpy(X_train[i:i+batch_size]).to(device)
            labels = torch.from_numpy(y_train[i:i+batch_size]).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            ######################### GRADNORM #########################
            if iter==0:
                weights = criterion.loss_weights
                T = weights.sum().detach() # sum of weights
                # set gradnorm optimizer
                gradnorm_optimizer = torch.optim.Adam([weights], lr=gradnorm_learning_rate)
                previous_losses = criterion.losses
            # clear gradients of model
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            losses = criterion.losses
            # compute L2 norm of the gradients for each task
            gw = []
            for i in range(criterion.task_num):
                dl = torch.autograd.grad(weights[i]*losses[i], weights, retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)
            # compute loss ratio per task
            losses_ratio = torch.div(torch.stack(losses), torch.stack(previous_losses))
            # compute relative inverse training rate per task
            rt = losses_ratio / losses_ratio.mean()
            # compuate average gradient norm
            gw_avg = gw.mean().detach()
            # compute gradnorm loss
            constant = (gw_avg * rt ** gradnorm_alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            # clear gradients of weights
            gradnorm_optimizer.zero_grad()
            # backward pass
            gradnorm_loss.backward()
            if logging_weights:
                caches['weights'].append(weights.detach().cpu().numpy().copy())
                caches['loss_ratio'].append(losses_ratio.detach().cpu().numpy().copy())
            # update model weights
            optimizer.step()
            # update gradnorm loss weights
            gradnorm_optimizer.step()
            # renormalize weights
            weights = (weights / weights.sum() * T).detach()
            weights = torch.nn.Parameter(weights)
            gradnorm_optimizer = torch.optim.Adam([weights], lr=gradnorm_learning_rate)
            # update iter
            iter += 1
            ################################################################
            train_loss += loss.item()
            caches['losses'].append(losses.detach().cpu().numpy().copy())

        train_loss /= (X_train.shape[0] / batch_size)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                inputs = torch.from_numpy(X_val[i:i+batch_size]).to(device)
                labels = torch.from_numpy(y_val[i:i+batch_size]).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= (X_val.shape[0] / batch_size)
        val_losses.append(val_loss)
        # Print
        print('Epoch: {} | Train loss: {:.4f} | Val loss: {:.4f}'.format(epoch, train_loss, val_loss))
    return model, train_losses, val_losses