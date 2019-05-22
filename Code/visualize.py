import os
from argparse import ArgumentParser
import random
import numpy as np

from models import VanillaCNN, train_model, eval_model, test_model
from crnn import CRNN, CRNNNoLSTM

import torch
from torch.optim import Adam
from torchvision import models

import matplotlib.pyplot as plt
from misc_functions import preprocess_image, recreate_image, save_image

"""
Created on Sat Nov 18 23:12:08 2017

Class author: Utku Ozbulak - github.com/utkuozbulak
Implementation by camille noufi 5/14/2019
"""


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, in_channels, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.in_channels = in_channels
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (172, 80, self.in_channels)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        print("hook")
        print(processed_image.shape)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        nlayers = len(self.model)
        for i in range(1, 3):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image every 5th iteration of backprop
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '_hook.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        
        print("Without Hooks:")
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (172, 80, self.in_channels)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        
        print(processed_image.shape)
        # Define optimizer for the image
        #nlayers = len(self.model)
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '_no-hook.jpg'
                save_image(self.created_image, im_path)


def load_model(device,filepath=None,params=None):
    if filepath is None:
        model = models.vgg16(pretrained=True).features
        nchan = 3
    else:
        model = VanillaCNN(params['kernel_size'], params['in_channels'], params['num_filters'], params['num_classes'], params['dropout_rate'])
        w = torch.load(model_file) #map_location = device
        model.load_state_dict(w)
        nchan = 1
    return (model, nchan)



def get_computing_device():
#     print('Pytorch CUDA test:')
#     print(torch.__version__)
#     print(torch.cuda.is_available())

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA GPU\n')
    else:
        print('Using CPU\n')
    
    return device



if __name__ == '__main__':

    #print available computing state
    device = get_computing_device()

    #visualization params:
    cnn_layer = 3
    filter_pos = 5

    #load desired model:
    params = {}
    params['kernel_size'] = 3
    params['in_channels'] = 1
    params['num_filters'] = 32
    params['dropout_rate'] = 0.5
    params['num_classes'] = 10

    #model_file = None #use for loading pretrained VGG16
    model_file = "trained_cnn_model_params.pt"
    (model,nchan) = load_model(device,model_file,params)
    
    print("Loaded Model:\n")
    print(model.conv1)

    
    
    #declare visualization instance
    lv = CNNLayerVisualization(model, nchan, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    #lv.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    lv.visualise_layer_without_hooks()
