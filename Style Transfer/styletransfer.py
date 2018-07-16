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
I have taken the idea of algorithmic flow from open source repository "https://github.com/cysmith/neural-style-tf"

"""

from config import get_config
import vgg19
import numpy as np
import tensorflow as tf
import scipy.misc
from tqdm import trange


def generate_random_image(content_img, noise_ratio):
    """ function to generate random images
    Parameters
    ----------
    content_img: Content image of size hxwxc
    noise_ratio: float having value between 0.0 to 1.0

    Return
    ------
    random_img: Randomly generated image of size hxwxc
    
    """
    #We have initialized the random image by adding noise to the content image
    # Randomly generated numbers between -20 to 20 and added that to each color value
    random_img= np.random.uniform(-20,20,(1,config.img_height,config.img_width,config.color_channels))
    random_img = random_img * noise_ratio + content_img*(1-noise_ratio)
    return  random_img


def content_loss(sess,model,content_img):
    """ function to calculate content loss 
    Parameters
    ----------
    sess: tensorflow session
    model: dictionary having CNN model
    content_img: content image of size hxwxc

    Return
    ------
    content_loss: scalar of content loss
    
    """

    
    _, h, w, c = content_img.shape
    #feed content image to model input
    sess.run( model['input'].assign(content_img))
    # Calculate the output for content image from content layer which is set in config file
    content_out = sess.run(model[config.content_layer])

    # Take the output for generated image later in the program we will feed the generated image into the model
    generated_out=model[config.content_layer]

    # Calclulate the content loss using squre -error loss
    content_loss = (1/(4*h*w*c))*tf.reduce_sum(tf.square(tf.subtract(content_out,generated_out)))
    
    return content_loss


def gram_matrix(F_maps):
    """ Function to calculate gram matrix, 
    Parameters
    ----------
    F_maps: ndarray, feature map of shape(c, h*w) 

    Return
    ------
    mat: Gram matrix of shape (c,c)
    
    """
    mat = tf.matmul(tf.transpose(F_maps),F_maps)
    return mat

def style_loss(sess,model,style_img):
    """ Function to calculate style loss for layers 
    Parameters
    ----------
    sess: tensorflow session
    model: dictionary having CNN model
    style_img: style image of size hxwxc 

    Return
    ------
    s_loss: scalar of style loss
    
    """

    #assign style image to model
    sess.run(model['input'].assign(style_img))
    s_loss=0

    #loop to calculate style loss of each style layer
    for layer, weight in zip(config.style_layers,config.style_layer_weights):
        s_layer_out = sess.run(model[layer])
        g_layer_out = model[layer]
        loss=style_layer_loss(s_layer_out,g_layer_out)
        s_loss += loss*weight
    return s_loss


def style_layer_loss(style_img,generated_img):
    _, h, w, c= style_img.shape
    

    M= h*w
    N= c
    style_img= tf.reshape(style_img,(M,c))
    generated_img = tf.reshape(generated_img, (M, c))
    #Gram matrix for style image
    style_mat=gram_matrix(style_img)
    #Gram matrix for generated image
    generated_mat=gram_matrix(generated_img)
    #calculate the style square error loss 
    style_loss= (1/(4*(N**2)*(M**2)))*tf.reduce_sum(tf.square(tf.subtract(style_mat,generated_mat)))
    return style_loss


def total_cost(content_loss, style_loss, alpha, beta):
    """ Function to calculate style loss for layer
    Parameters
    ----------
    content_loss: loss from content layer
    style_loss: loss from style layers
    alpha: content layer contribution cofficient
    beta: style layer contribution cofficient

    Return
    ------
    t_cost: Total loss
    
    """

    
    t_cost= alpha*content_loss+beta*style_loss
    return t_cost

def image_preprocess(img):
    """
    Parameters
    ----------
    img: image of size (h x w x C)

    Return
    ------
    img: preprocessed image
    """
    #reshape the image to match the input shape of VGG 19
    img_shp=img.shape
    shp=(1,img.shape[0],img_shp[1],img_shp[2])
    img=np.reshape(img,(shp))
    means=np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    img=img-means
    return img

def store_img(path,img):
    """
    Parameters
    ----------
    path: Path to store the image

    """
    means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    img = img + means
    
    image = np.clip(img[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def main(config):
    sess= tf.InteractiveSession()
    #read the content image
    content_img= scipy.misc.imread(config.content_dir)
    #preprocess content image
    content_img= image_preprocess(content_img)
    #read the style image
    style_img = scipy.misc.imread(config.style_dir)
    #preprocess the style image
    style_img= image_preprocess(style_img)

    #load the vgg19 mode;
    model = vgg19.load_vgg19(config.vgg_path,config)

    #calculate content loss
    c_loss= content_loss(sess,model,content_img)
    #calculate style loss
    s_loss= style_loss(sess,model,style_img)

    aplha = config.alpha
    beta = config.beta 

    #calculate total loss
    total_loss = total_cost(c_loss,s_loss,aplha,beta)
    #optimization
    optim= tf.train.AdamOptimizer(config.learning_rate)
    train= optim.minimize(total_loss)
    random_img=generate_random_image(content_img,config.noise_ratio)
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(random_img))

    #iteration to minimize the losses iteratively
    for idx in trange(config.max_itr):
        sess.run(train)
        #storing image after 20 iteration to see intermmidiate results 
        if(idx % 10 == 0):
            store_img(config.test_dir + str(idx) + '.png',sess.run(model['input']))

    #store the final result        
    store_img(config.output_dir, sess.run(model['input']))




if __name__ == '__main__':
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        #print_usage()

        exit(1)

    main(config)
