import torch
import torch.nn as nn


####################################################################################################
#####     MAKE BLOCKS FUNCTIONS   ##################################################################

def make_vgg_1d_block(input_channel, kernel_size, stride, padding, output_channel, repeat_conv=1, batch_norm=False, inplace=True, maxpool_kernel_size=4,
                      endpooling = 'maxpool'):
    layers = []
    for i in range(repeat_conv):
        layers.append(nn.Conv1d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding))
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_channel))
        layers.append(nn.ReLU(inplace=inplace))
        input_channel = output_channel
    if endpooling=='maxpool': layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding=1))
    else: layers.append(nn.AvgPool1d(kernel_size=kernel_size, stride=2, padding=1))
    return nn.Sequential(*layers)


def make_echo_vgg_backbone(input_echo_length, kernel_sizes, output_channels, strides, paddings, repeat_convs, batch_norm=True, maxpool_kernel_sizes=None,
                           input_channel=1,
                           endpooling='maxpool'):
    layers = []
    current_input_channel = input_channel
    if maxpool_kernel_sizes==None: maxpool_kernel_sizes = (4,)*len(kernel_sizes)
    for i in range(len(kernel_sizes)):
        if endpooling=='avgpool' and i==len(kernel_sizes)-1:
            layers.append(make_vgg_1d_block(input_channel=current_input_channel, kernel_size=kernel_sizes[i], output_channel=output_channels[i], stride=strides[i],
                                        padding=paddings[i], repeat_conv=repeat_convs[i], batch_norm=batch_norm, maxpool_kernel_size=maxpool_kernel_sizes[i],
                                            endpooling='avgpool'))
        else: layers.append(make_vgg_1d_block(input_channel=current_input_channel, kernel_size=kernel_sizes[i], output_channel=output_channels[i], stride=strides[i],
                                        padding=paddings[i], repeat_conv=repeat_convs[i], batch_norm=batch_norm, maxpool_kernel_size=maxpool_kernel_sizes[i],))
        
        current_input_channel = output_channels[i]
        input_echo_length = int(input_echo_length/2)
    return nn.Sequential(*layers), input_echo_length

def make_fc_regression_head(input_units, hidden_layers, output_units, dropout=False, dropout_rate=0.1):
    layers = []
    layers.append(nn.Linear(input_units, hidden_layers[0]))
    layers.append(nn.ReLU())
    if dropout:
        layers.append(nn.Dropout(dropout_rate))
    for i in range(len(hidden_layers)-1):
        layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(hidden_layers[-1], output_units))
    return nn.Sequential(*layers)

####################################################################################################
##------------------------------------------------------------------------------------------------##
####################################################################################################
#####     SINGLE TASK MODELS    ####################################################################

class EchoVGG_regression(nn.Module):
    def __init__(self, input_echo_length,
                kernel_sizes = (7,7,5,5,3,3),
                output_channels = (16,32,64,64,64,64),
                strides = (1,1,1,1,1,1),
                paddings = (3,3,2,2,1,1),
                repeat_convs = (2,2,3,3,4,4),
                maxpool_kernel_sizes = (8,8,4,4,3,3),
                endpooling = 'maxpool',
                fc_hidden_layers = (512,128),
                fc_output_units = 1,
                batch_norm=True,
                dropout=False,
                dropout_rate=0.05):
        super(EchoVGG_regression, self).__init__()
        self.backbone, input_echo_length = make_echo_vgg_backbone(input_echo_length, kernel_sizes, output_channels, strides, paddings, repeat_convs,
                                                                  batch_norm=batch_norm,maxpool_kernel_sizes=maxpool_kernel_sizes, endpooling=endpooling, input_channel=1)
        self.regression_head = make_fc_regression_head(2*input_echo_length*output_channels[-1], fc_hidden_layers, fc_output_units, dropout=dropout, dropout_rate=dropout_rate)

    def forward(self, x):
        x_left, x_right = x[:,0,:].unsqueeze(1), x[:,1,:].unsqueeze(1)
        x_left = self.backbone(x_left)
        x_right = self.backbone(x_right)
        x_left = x_left.reshape(x_left.size(0), -1)
        x_right = x_right.reshape(x_right.size(0), -1)
        x = torch.cat((x_left, x_right), dim=1)
        x = self.regression_head(x)
        return x
    
class UniEchoVGG_regression(nn.Module):
    def __init__(self,input_echo_length,
                kernel_sizes = (7,7,5,5,3,3),
                output_channels= (16,32,64,128,128,128),
                strides = (1,1,1,1,1,1),
                paddings = (3,2,2,2,1,1),
                repeat_convs = (2,2,3,3,4,4),
                maxpool_kernel_sizes = (8,8,4,4,3,3),
                endpooling = 'maxpool',
                mixing_layer_units = None,
                fc_hidden_layers = (512,128),
                fc_output_units = 1,
                batch_norm=True,
                dropout=True,
                dropout_rate=0.05):
        super(UniEchoVGG_regression, self).__init__()
        self.backbone, input_echo_length = make_echo_vgg_backbone(input_echo_length, kernel_sizes, output_channels, strides, paddings, repeat_convs,
                                                                  batch_norm=batch_norm,maxpool_kernel_sizes=maxpool_kernel_sizes, endpooling=endpooling, input_channel=2)
        self.regression_head = make_fc_regression_head(input_echo_length*output_channels[-1], fc_hidden_layers, fc_output_units, dropout=dropout, dropout_rate=dropout_rate)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.regression_head(x)
        return x

####################################################################################################
##------------------------------------------------------------------------------------------------##
####################################################################################################
#####     MULTI TASK MODELS    #####################################################################

class EchoVGG_PoseEstimator(nn.Module):
    def __init__(self, input_echo_length,
                kernel_sizes = (7,7,5,5,3,3),
                output_channels = (16,32,64,64,64,64),
                strides = (1,1,1,1,1,1),
                paddings = (3,3,2,2,1,1),
                repeat_convs = (2,2,3,3,4,4),
                maxpool_kernel_sizes = (8,8,4,4,3,3),
                endpooling = 'maxpool',
                mixing_layer_units = None,
                distance_hidden_layers = (512,128),
                azimuth_hidden_layers = (512,128),
                orientation_hidden_layers = (512,128),
                batch_norm=True,
                dropout=False,
                dropout_rate=0.05):
        super(EchoVGG_PoseEstimator, self).__init__()
        self.backbone, input_echo_length = make_echo_vgg_backbone(input_echo_length, kernel_sizes, output_channels, strides, paddings, repeat_convs,
                                                                  batch_norm=batch_norm,maxpool_kernel_sizes=maxpool_kernel_sizes, endpooling=endpooling, input_channel=1)
        if mixing_layer_units:
            self.mixing_layer = nn.Sequential(nn.Linear(2*input_echo_length*output_channels[-1], mixing_layer_units), nn.ReLU())
            input_echo_length = int(mixing_layer_units/2)
        self.distance_head = make_fc_regression_head(input_units=2*input_echo_length*output_channels[-1], hidden_layers=distance_hidden_layers, output_units=1, dropout=dropout, dropout_rate=dropout_rate)
        self.azimuth_head = make_fc_regression_head(input_units=2*input_echo_length*output_channels[-1], hidden_layers=azimuth_hidden_layers, output_units=1, dropout=dropout, dropout_rate=dropout_rate)
        self.orientation_head = make_fc_regression_head(input_units=2*input_echo_length*output_channels[-1], hidden_layers=orientation_hidden_layers, output_units=1, dropout=dropout, dropout_rate=dropout_rate)

    def forward(self, x):
        x_left, x_right = x[:,0,:].unsqueeze(1), x[:,1,:].unsqueeze(1)
        x_left = self.backbone(x_left)
        x_right = self.backbone(x_right)
        x_left = x_left.reshape(x_left.size(0), -1)
        x_right = x_right.reshape(x_right.size(0), -1)
        x = torch.cat((x_left, x_right), dim=1)
        if hasattr(self, 'mixing_layer'): x = self.mixing_layer(x)
        distance = self.distance_head(x)
        azimuth = self.azimuth_head(x)
        orientation = self.orientation_head(x)
        return distance, azimuth, orientation
    

class UniEchoVGG_PoseEstimator(nn.Module):
    def __init__(self, input_echo_length,
                 kernel_sizes = (7,7,5,5,3,3),
                output_channels = (16,32,64,64,64,64),
                strides = (1,1,1,1,1,1),
                paddings = (3,3,2,2,1,1),
                repeat_convs = (2,2,3,3,4,4),
                maxpool_kernel_sizes = (8,8,4,4,3,3),
                endpooling = 'maxpool',
                mixing_layer_units = None,
                distance_hidden_layers = (512,128),
                azimuth_hidden_layers = (512,128),
                orientation_hidden_layers = (512,128),
                batch_norm=True,
                dropout=False,
                dropout_rate=0.05):
        super(UniEchoVGG_PoseEstimator, self).__init__()
        self.backbone, input_echo_length = make_echo_vgg_backbone(input_echo_length, kernel_sizes, output_channels, strides, paddings, repeat_convs,
                                                                  batch_norm=batch_norm,maxpool_kernel_sizes=maxpool_kernel_sizes, endpooling=endpooling, input_channel=1)
        if mixing_layer_units:
            self.mixing_layer = nn.Sequential(nn.Linear(input_echo_length*output_channels[-1], mixing_layer_units), nn.ReLU())
            input_echo_length = mixing_layer_units
        self.distance_head = make_fc_regression_head(input_units=input_echo_length*output_channels[-1], hidden_layers=distance_hidden_layers, output_units=1, dropout=dropout, dropout_rate=dropout_rate)
        self.azimuth_head = make_fc_regression_head(input_units=input_echo_length*output_channels[-1], hidden_layers=azimuth_hidden_layers, output_units=1, dropout=dropout, dropout_rate=dropout_rate)
        self.orientation_head = make_fc_regression_head(input_units=input_echo_length*output_channels[-1], hidden_layers=orientation_hidden_layers, output_units=1, dropout=dropout, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        if hasattr(self, 'mixing_layer'): x = self.mixing_layer(x)
        distance = self.distance_head(x)
        azimuth = self.azimuth_head(x)
        orientation = self.orientation_head(x)
        return distance, azimuth, orientation
    
