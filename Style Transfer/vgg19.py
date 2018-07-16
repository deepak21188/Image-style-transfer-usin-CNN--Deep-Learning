"""
This program implements the image style transfer using convlution neural network
    Copyright (C) 2018  Deepak Kumar

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
"""
I have taken the idea of how to use pre-trained VGG network from open source repository "https://github.com/cysmith/neural-style-tf"
"""

import scipy.io
import scipy.misc
import tensorflow as tf
import numpy as np


def load_vgg19(path, config):
    """ Function to laod vgg19 model
    Parameters
    ----------
    path: path of vgg model file
    config: config file reference 

    Return
    ------
    model: pyhton dictionary having vgg 19 model
    
    """
    vgg_model= scipy.io.loadmat(path)
    vgg_layers= vgg_model['layers'][0]

    model={}
    model['input'] = tf.Variable(np.zeros((1, config.img_height, config.img_width, config.color_channels), dtype=np.float32))
    print("input shape ({},{},{},{})".format(1,config.img_height,config.img_width,config.color_channels))

    model['conv1_1'] = conv2d_layer('conv1_1', model['input'],get_weights(vgg_layers, 0),get_bias(vgg_layers, 0))
    model['relu1_1'] = relu_layer('relu1_1', model['conv1_1'] )

    model['conv1_2'] = conv2d_layer('conv1_2', model['relu1_1'], get_weights(vgg_layers, 2),get_bias(vgg_layers, 2))
    model['relu1_2'] = relu_layer('relu1_2', model['conv1_2'])

    model['pool1'] = pool_layer('pool1', model['relu1_2'],config)

    model['conv2_1'] = conv2d_layer('conv2_1', model['pool1'], get_weights(vgg_layers, 5),get_bias(vgg_layers, 5))
    model['relu2_1'] = relu_layer('relu2_1', model['conv2_1'])

    model['conv2_2'] = conv2d_layer('conv2_2', model['relu2_1'],get_weights(vgg_layers, 7),get_bias(vgg_layers, 7))
    model['relu2_2'] = relu_layer('relu2_2', model['conv2_2'])

    model['pool2'] = pool_layer('pool2', model['relu2_2'],config)


    model['conv3_1'] = conv2d_layer('conv3_1', model['pool2'], get_weights(vgg_layers, 10),get_bias(vgg_layers, 10))
    model['relu3_1'] = relu_layer('relu3_1', model['conv3_1'])

    model['conv3_2'] = conv2d_layer('conv3_2', model['relu3_1'],get_weights(vgg_layers, 12),get_bias(vgg_layers, 12))
    model['relu3_2'] = relu_layer('relu3_2', model['conv3_2'])

    model['conv3_3'] = conv2d_layer('conv3_3', model['relu3_2'], get_weights(vgg_layers, 14),get_bias(vgg_layers, 14))
    model['relu3_3'] = relu_layer('relu3_3', model['conv3_3'])

    model['conv3_4'] = conv2d_layer('conv3_4', model['relu3_3'],get_weights(vgg_layers, 16),get_bias(vgg_layers, 16))
    model['relu3_4'] = relu_layer('relu3_4', model['conv3_4'])

    model['pool3'] = pool_layer('pool3', model['relu3_4'],config)


    model['conv4_1'] = conv2d_layer('conv4_1', model['pool3'],get_weights(vgg_layers, 19),get_bias(vgg_layers, 19))
    model['relu4_1'] = relu_layer('relu4_1', model['conv4_1'])

    model['conv4_2'] = conv2d_layer('conv4_2', model['relu4_1'],get_weights(vgg_layers, 21),get_bias(vgg_layers, 21))
    model['relu4_2'] = relu_layer('relu4_2', model['conv4_2'])

    model['conv4_3'] = conv2d_layer('conv4_3', model['relu4_2'],get_weights(vgg_layers, 23),get_bias(vgg_layers, 23))
    model['relu4_3'] = relu_layer('relu4_3', model['conv4_3'])

    model['conv4_4'] = conv2d_layer('conv4_4', model['relu4_3'], get_weights(vgg_layers, 25),get_bias(vgg_layers, 25))
    model['relu4_4'] = relu_layer('relu4_4', model['conv4_4'])

    model['pool4'] = pool_layer('pool4', model['relu4_4'],config)


    model['conv5_1'] = conv2d_layer('conv5_1', model['pool4'],get_weights(vgg_layers, 28),get_bias(vgg_layers, 28))
    model['relu5_1'] = relu_layer('relu5_1', model['conv5_1'])

    model['conv5_2'] = conv2d_layer('conv5_2', model['relu5_1'], get_weights(vgg_layers, 30),get_bias(vgg_layers, 30))
    model['relu5_2'] = relu_layer('relu5_2', model['conv5_2'])

    model['conv5_3'] = conv2d_layer('conv5_3', model['relu5_2'], get_weights(vgg_layers, 32),get_bias(vgg_layers, 32))
    model['relu5_3'] = relu_layer('relu5_3', model['conv5_3'])

    model['conv5_4'] = conv2d_layer('conv5_4', model['relu5_3'],get_weights(vgg_layers, 34),get_bias(vgg_layers, 34))
    model['relu5_4'] = relu_layer('relu5_4', model['conv5_4'])

    model['pool5'] = pool_layer('pool5', model['relu5_4'],config)

    return model



def get_weights(layers, layer_num):
    #function to get weights at perticual layer
    w= layers[layer_num][0][0][2][0][0]
    W= tf.constant(w)
    return W

def get_bias(layers, layer_num):
    #function to get bais at perticual layer
    b= layers[layer_num][0][0][2][0][1]
    B= tf.constant(np.reshape(b,(len(b))))
    return B

def conv2d_layer(layer_name, input, W,B):
    #function to get convolution layer
    conv_output= tf.nn.conv2d(input,filter= W,strides=[1, 1, 1, 1], padding='SAME')+B
    print("--{} | layer_output: {} | filters : {}".format(layer_name,conv_output.get_shape(),W.get_shape()))
    return conv_output

def relu_layer(layer_name, input):
    #function to get relu layer
    relu_output= tf.nn.relu(input)
    print("--{} | layer_output: {}".format(layer_name, relu_output.get_shape()))
    return relu_output

def pool_layer(layer_name , input, config):
    #function to get pooling layer
    if config.pool_type =="avg":
       pool_output = tf.nn.avg_pool(input, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME')

    elif config.pool_type == "max":
        pool_output = tf.nn.max_pool(input, ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1], padding='SAME')

    print('--{}   | shape={}'.format(layer_name, pool_output.get_shape()))
    return pool_output
















