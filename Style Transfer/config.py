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


import argparse


arg = argparse.ArgumentParser(description="Neural style transfer")

arg.add_argument("--img_width", type=int,
                       default=300,
                       help="Input image width")
arg.add_argument("--img_height", type=int,
                       default=200,
                       help="Input image height")
arg.add_argument("--color_channels", type=int,
                       default=3,
                       help="Number of color channels in input image")
arg.add_argument("--vgg_path", type=str,
                       default="G:\H Drive\CSC586B Deep learning\Project Practice\imagenet-vgg-verydeep-19.mat",
                       help="Directory where vgg model is stored")
arg.add_argument("--output_dir", type=str,
                       default="Images/ouput.png",
                       help="Directory where output image will be stored")
arg.add_argument("--content_dir", type=str,
                       default="Images/content.jpg",
                       help="Directory where content image is stored")
arg.add_argument("--style_dir", type=str,
                       default="Images/style.jpg",
                       help="Directory where style image is stored")
arg.add_argument("--test_dir", type=str,
                       default="Images/",
                       help="Directory where test images are stored")
arg.add_argument("--pool_type", type=str,
                       default="avg",
                       choices=["avg","max"],
                       help="Directory where output image will be stored")
arg.add_argument("--noise_ratio", type=float,
                       default="0.5",
                       help="Noise ratio by randomly generated image")
arg.add_argument('--content_layer', type=str,
                        default='conv4_2',
                        help='VGG layer used for content iange')
arg.add_argument('--style_layers', type=str,
    default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
    help='VGG19 layers used for the style image')
arg.add_argument('--content_layer_weight', type=float,
                    default=1.0,
                    help='weights of each content layer to loss')
arg.add_argument('--style_layer_weights', type=float,
                    default=[0.2, 0.2, 0.2, 0.2, 0.2],
                    help='weights of each style layer to loss')
arg.add_argument('--alpha', type=int,
                    default=5e0,
                    help='content layer contribution cofficient')
arg.add_argument('--beta', type=int,
                    default=1e4,
                    help='style layer contribution cofficient')
arg.add_argument('--learning_rate', type=float,
                    default=2e0,
                    help='Learning rate parameter for the Adam optimizer.')
arg.add_argument('--max_itr', type=int,
                    default=1000,
                    help='Max number of iterations for optimization process')


def get_config():
    config,unparsed = arg.parse_known_args()
    return config,unparsed


