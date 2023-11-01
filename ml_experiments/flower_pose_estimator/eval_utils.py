import numpy as np
import sys
import os
from matplotlib import pyplot as plt

from Sensors.FlowerEchoSimulator.Spatializer import wrapToPi
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.utils import shuffle


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

class WrapToPiMAELoss2(nn.Module): # using atan2 instead of fmod
    def __init__(self):
        super(WrapToPiMAELoss2, self).__init__()
        self.name = 'modpi_MAE2'
    def forward(self, prediction, target):
        return torch.mean(torch.abs(torch.atan2(torch.sin( target - prediction), torch.cos(target - prediction))))

class WrapToPiRMSELoss2(nn.Module): # using atan2 instead of fmod
    def __init__(self):
        super(WrapToPiRMSELoss2, self).__init__()
        self.name = 'modpi_RMSE2'
    def forward(self, prediction, target):
        return torch.sqrt(torch.mean((torch.atan2(torch.sin( target - prediction), torch.cos(target - prediction)))**2))

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


### Not Using ###shuffle
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



def trainSoftAdapt(model, X_train_, y_dis_train_, y_azi_train_, y_ori_train_, 
        X_val_, y_dis_val_, y_azi_val_, y_ori_val_, dis_loss_fn, azi_loss_fn, ori_loss_fn,
        epochs, batch_size, learning_rate, weight_decay, device,
        transform = None, val_transform = None,
        softadapt_on = True, softadapt_lookback_window = 20,
        schedule_weighting = False,
        schedule_weighting_epoch_steps = 20,
        schedule_weighting_min = 0.2,
        schedule_weighting_icrement = 0.1,
        custom_weight_decays = None,
        init_loss_weights = None, init_softmax = True,
        swith_to_SGD=True, switching_epoch=50, lr_scheduler_threshold=1e-3, lr_scheduler_patience=10, lr_scheduler_factor=0.1,
        SGD_lr=1e-2,  momentum=0.92):
    if softadapt_on and schedule_weighting: raise ValueError('Cannot use both softadapt and schedule_weighting')
    model = model.to(device)
    criterions = [dis_loss_fn, azi_loss_fn, ori_loss_fn]
    if custom_weight_decays:
        models_custom_params = []
        models_custom_params.append({'params': model.backbone.parameters(), 'weight_decay': custom_weight_decays['backbone']})
        if hasattr(model, 'fc_mixing'): models_custom_params.append({'params': model.fc_mixing.parameters(), 'weight_decay': custom_weight_decays['mixing']})
        models_custom_params.append({'params': model.fc_distance.parameters(), 'weight_decay': custom_weight_decays['distance']})
        models_custom_params.append({'params': model.fc_azimuth.parameters(), 'weight_decay': custom_weight_decays['azimuth']})
        models_custom_params.append({'params': model.fc_orientation.parameters(), 'weight_decay': custom_weight_decays['orientation']})
        optimizer = optim.Adam(models_custom_params, lr=learning_rate,)
    else: optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    SoftAdaptSoftmax = nn.Softmax(dim=-1).to(device)
    if init_loss_weights == None:
        loss_weights = torch.ones(3).to(device)
    else: 
        loss_weights = torch.tensor(init_loss_weights).to(device)
    if init_softmax: loss_weights = SoftAdaptSoftmax(loss_weights)
    #previous_weights = loss_weights.detach().clone()
    previous_losses = None
    loss_record = deque(maxlen=softadapt_lookback_window)
    iter = 0
    train_losses = []
    val_losses = []

    cache = {'distance_train_loss': [], 'distance_val_loss': [], 'distance_loss_weight': [],
             'azimuth_train_loss': [], 'azimuth_val_loss': [], 'azimuth_loss_weight': [],
             'orientation_train_loss': [], 'orientation_val_loss': [], 'orientation_loss_weight': [],
    }
    for epoch in range(epochs):
        if swith_to_SGD and epoch == switching_epoch:
            optimizer = optim.SGD(model.parameters(), lr=SGD_lr, momentum=momentum, weight_decay=weight_decay)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=lr_scheduler_threshold, 
                                                                patience=lr_scheduler_patience, factor=lr_scheduler_factor, verbose=True)

        if schedule_weighting:
            if epoch % schedule_weighting_epoch_steps == 0:
                if loss_weights[0] > schedule_weighting_min:
                    loss_weights[0] -= (schedule_weighting_icrement/2)
                    loss_weights[1] -= (schedule_weighting_icrement/2)
                    loss_weights[2] += schedule_weighting_icrement
        # Training
        model.train()
        train_loss = 0
        individual_train_loss = [0,0,0]
        X_train_, y_dis_train_, y_azi_train_, y_ori_train_ = shuffle(X_train_, y_dis_train_, y_azi_train_, y_ori_train_)
        for i in range(0, X_train_.shape[0], batch_size):
            if transform: inputs = transform(X_train_[i:i+batch_size]).to(device)
            else:inputs = torch.from_numpy(X_train_[i:i+batch_size]).to(device)
            labels = []
            labels.append(torch.from_numpy(y_dis_train_[i:i+batch_size]).to(device))
            labels.append(torch.from_numpy(y_azi_train_[i:i+batch_size]).to(device))
            labels.append(torch.from_numpy(y_ori_train_[i:i+batch_size]).to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = []
            for i, (label, criterion) in enumerate(zip(labels, criterions)):
                losses.append(criterion(outputs[i], label))
                individual_train_loss[i] += losses[-1].item()
            # calculate loss as weighted sum of losses
            loss = torch.dot(loss_weights, torch.stack(losses))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            ########################   SOFT ADAPT   #################################
            if softadapt_on:
                if iter < softadapt_lookback_window:
                    loss_record.append(losses)
                else:
                    current_losses = torch.stack(losses).detach().clone()
                    avg_previous_losses = torch.mean(torch.stack([torch.stack(subloss).detach().clone() for subloss in loss_record]), axis=0)
                    mu = torch.max(torch.div(current_losses, (avg_previous_losses + 1e-8)))
                    loss_weights = SoftAdaptSoftmax(torch.div(current_losses, (avg_previous_losses + 1e-8)) - mu)
                    loss_record.append(losses)
                iter += 1
                cache['distance_loss_weight'].append(loss_weights[0].item())
                cache['azimuth_loss_weight'].append(loss_weights[1].item())
                cache['orientation_loss_weight'].append(loss_weights[2].item())
            ########################################################################
            # if softadapt_on:
            #     if previous_losses == None: # skip the first iteration, set previous_losses
            #         previous_losses = torch.stack(losses).detach().clone()
            #     else: # perfrom the update for the loss weights in later iterations
            #         current_losses = torch.stack(losses).detach().clone()
            #         mu = torch.max(torch.div(current_losses, (previous_losses + 1e-8)))
            #         loss_weights = SoftAdaptSoftmax(torch.div(current_losses, (previous_losses + 1e-8)) - mu)
            #         previous_losses = current_losses.clone()
            ########################################################################

        train_loss /= (X_train_.shape[0] / batch_size)
        train_losses.append(train_loss)
        for k in range(3): individual_train_loss[k] /= (X_train_.shape[0] / batch_size)
        cache['distance_train_loss'].append(individual_train_loss[0])
        cache['azimuth_train_loss'].append(individual_train_loss[1])
        cache['orientation_train_loss'].append(individual_train_loss[2])
        # Validation
        model.eval()
        val_loss = 0
        individual_val_loss = [0,0,0]
        with torch.no_grad():
            for i in range(0, X_val_.shape[0], batch_size):
                if val_transform: inputs = val_transform(X_val_[i:i+batch_size]).to(device)
                else:inputs = torch.from_numpy(X_val_[i:i+batch_size]).to(device)
                labels = []
                labels.append(torch.from_numpy(y_dis_val_[i:i+batch_size]).to(device))
                labels.append(torch.from_numpy(y_azi_val_[i:i+batch_size]).to(device))
                labels.append(torch.from_numpy(y_ori_val_[i:i+batch_size]).to(device))
                outputs = model(inputs)
                losses = []
                for i, (label, criterion) in enumerate(zip(labels, criterions)):
                    losses.append(criterion(outputs[i], label))
                    individual_val_loss[i] += losses[-1].item()
                loss = torch.dot(loss_weights, torch.stack(losses))
                val_loss += loss.item()
        val_loss /= (X_val_.shape[0] / batch_size)
        val_losses.append(val_loss)        
        if swith_to_SGD and epoch >= switching_epoch: lr_scheduler.step(val_loss)
        for k in range(3): individual_val_loss[k] /= (X_val_.shape[0] / batch_size)
        cache['distance_val_loss'].append(individual_val_loss[0])
        cache['azimuth_val_loss'].append(individual_val_loss[1])
        cache['orientation_val_loss'].append(individual_val_loss[2])
        # Print
        print('Epoch: {}, lr={}, lw=[{:.2f},{:.2f},{:.2f}] | Losses = train: {:.4f} [{:.3f}, {:.3f},{:.3f}] | val: {:.4f}[{:.3f}, {:.3f},{:.3f}]'\
              .format(epoch, optimizer.param_groups[0]['lr'], loss_weights[0].item(), loss_weights[1].item(), loss_weights[2].item(), \
                      train_loss, cache['distance_train_loss'][-1], cache['azimuth_train_loss'][-1], cache['orientation_train_loss'][-1],\
                      val_loss, cache['distance_val_loss'][-1], cache['azimuth_val_loss'][-1], cache['orientation_val_loss'][-1]))
    return model, train_losses, val_losses, loss_weights, cache

def testMTL(model, X_test_, y_dis_test_, y_azi_test_, y_ori_test_, loss_weights, dis_loss_fn, azi_loss_fn, ori_loss_fn, batch_size, device, transform=None):
    model = model.to(device)
    model.eval()
    criterions = [dis_loss_fn, azi_loss_fn, ori_loss_fn]
    test_loss = 0
    individual_test_loss = [0,0,0]
    with torch.no_grad():
        for i in range(0, X_test_.shape[0], batch_size):
            if transform: inputs = transform(X_test_[i:i+batch_size]).to(device)
            else:inputs = torch.from_numpy(X_test_[i:i+batch_size]).to(device)
            labels = []
            labels.append(torch.from_numpy(y_dis_test_[i:i+batch_size]).to(device))
            labels.append(torch.from_numpy(y_azi_test_[i:i+batch_size]).to(device))
            labels.append(torch.from_numpy(y_ori_test_[i:i+batch_size]).to(device))
            outputs = model(inputs)
            losses = []
            for i, (label, criterion) in enumerate(zip(labels, criterions)):
                losses.append(criterion(outputs[i], label))
                individual_test_loss[i] += losses[-1].item()
            loss = torch.dot(loss_weights, torch.stack(losses))
            test_loss += loss.item()
    test_loss /= (X_test_.shape[0] / batch_size)
    for k in range(3): individual_test_loss[k] /= (X_test_.shape[0] / batch_size)
    return test_loss, individual_test_loss

def predict(model, X, batch_size, device, transform=None):
    model = model.to(device)
    model.eval()
    y_pred = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            if transform: inputs = transform(X[i:i+batch_size]).to(device)
            else:inputs = torch.from_numpy(X[i:i+batch_size]).to(device)
            outputs = model(inputs)
            if i == 0:
                for output in outputs: y_pred.append(output.cpu().numpy())
            else:
                for k, output in enumerate(outputs): y_pred[k] = np.vstack((y_pred[k], output.cpu().numpy()))
    return y_pred

def save_model(model, model_name, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, model_name))

def load_model(model, model_name, model_path):
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    return model

def save_losses(train_losses, val_losses, model_name, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    np.save(os.path.join(model_path, '{}_train_losses.npy'.format(model_name)), np.asarray(train_losses))
    np.save(os.path.join(model_path, '{}_val_losses.npy'.format(model_name)), np.asarray(val_losses))

def load_losses(model_name, model_path):
    train_losses = np.load(os.path.join(model_path, '{}_train_losses.npy'.format(model_name)))
    val_losses = np.load(os.path.join(model_path, '{}_val_losses.npy'.format(model_name)))
    return train_losses, val_losses




class UniEchoVGG_GradCAM_heatmap:
    def __init__(self, model,
                 distance_loss_fn, azimuth_loss_fn, orientation_loss_fn,
                 transform, device='cuda:0'):
        self._check_model_compatiblity(model)
        self.model = model
        self.distance_loss_fn = distance_loss_fn
        self.azimuth_loss_fn = azimuth_loss_fn
        self.orientation_loss_fn = orientation_loss_fn
        self.transform = transform

        self.model.eval()
        self.model.to(device)
        self.device = device
        self.previous_prediction = None
        

    def run(self, inputs, targets, mode='all',):
        if self.transform:
            inputs = self.transform(inputs).float().to(self.device)
        else: inputs = torch.from_numpy(inputs).float().to(self.device)
        pred = self.model(inputs)
        loss1 = self.distance_loss_fn(pred[0], torch.from_numpy(targets[0]).float().to(self.device))
        loss2 = self.azimuth_loss_fn(pred[1], torch.from_numpy(targets[1]).float().to(self.device))
        loss3 = self.orientation_loss_fn(pred[2], torch.from_numpy(targets[2]).float().to(self.device))
        if mode=='all': loss =  loss1 + loss2 + loss3
        elif mode=='distance': loss = loss1
        elif mode=='azimuth': loss = loss2
        elif mode=='orientation': loss = loss3
        else: raise ValueError('mode must be one of all, distance, azimuth, orientation but got {}'.format(mode))
        loss.backward()

        grads = self.model.get_activations_gradient()
        pooled_grads = torch.mean(grads, dim=[0, 2])
        activations = self.model.get_activations(inputs).detach()
        for i in range(activations.shape[1]):
            activations[:, i, :] *= pooled_grads[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap.detach())

        self.previous_prediction = pred
        return heatmap.cpu().numpy()


    def __call__(self, *args, **kwargs,):
        return self.run(*args, **kwargs)
    

    def _check_model_compatiblity(self, model):
        if not hasattr(model, 'get_activations_gradient'):
            raise ValueError('Model {} does not have get_activations_gradient method defined.'.format(model.__class__.__name__))
        if not hasattr(model, 'get_activations'):
            raise ValueError('Model {} does not have get_activations method defined.'.format(model.__class__.__name__))


def upsample_heatmap(heatmap_numpy):
    #heatmap_numpy = np.flip(heatmap_numpy)
    upsample = np.zeros(512,)
    for i in range(heatmap_numpy.shape[0]):
        upsample[i*34:(i+1)*34] = heatmap_numpy[i]
    return upsample