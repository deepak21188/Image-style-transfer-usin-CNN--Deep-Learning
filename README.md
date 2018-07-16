# Step-by-step guide for "Image Style Transfer Using Convolutional Neural Networks"
* Project folder contains three code files
    1. config.py
    2. styletransfer.py
    3. vgg19.py
* config.py file contains all the settings and hyperparameter as we have used in course assignments
* vgg19.py file contains all the functions to load and build model
* styletransfer.py file contains main function and code for calculating losses and optimization
* I have VGG19 model on my google drive and can be downloaded at [VGG19 Link](https://drive.google.com/file/d/1no_STC6Hldml1lD3gcso1_Tl_g99JW7_/view?usp=sharing) 
* Set the download vgg19 path to --vgg_path in cofiguration while running the program.
* There is a folder /Images inside project directory. It contains sample content images "content.jpg" and style image "style.jpg".
* Default path(Images folder in project directory) of content and style image is set in config file.
* Both content and style images should be of same size. Default Height and width values are set in the config file and can be changed in config 
* I have tested images for size (500x300) and (300x200)
* Once all the hyperparameters are set, we can run styletransfer.py.
* I have conducted experiments with different content and style images and with different hyper parameters like different noise_ratio value, different content_layers(conv1_2, conv2_2, conv4_4), different values of α/β, diffrent learning rate