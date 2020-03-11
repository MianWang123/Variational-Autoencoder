## Project Title

Generative Models: Convolutional Autoencoders and Variational Autoencoders

### Task and Model

The task is to generate as similar images (fake images) as the given dataset (real images).  
The Convolutional AutoEncoder(CAE) and Variational AutoEncoder(VAE) are created here, to reconstruct images for the UT Zappos50K Dataset.


### Prerequisites

I uploaded the zipped dataset(625 Mb) to Google drive, and used Google Colab to load & unzip the data, the dataset can be found here https://drive.google.com/file/d/1nYEgytPOkFyUjDQfBGzwCQbszf6OE143/view?usp=sharing. You can also upload the unzipped dataset directly to your Colab, just remember to change the data path in Step 1 --'DATASETS/UTZappos50K'.


### Introduction

The UT Zappos50K Dataset used here contains 4 types of images, i.e. Boots, Sandals, Shoes, Slippers.


### Data Visualization
For CAE, the training loss looks like:   
<img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/cae_loss.PNG" width='200'/>  
Besides, the outcomes of CAE can be seen below:
<figure class="half">
<img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/cae_pic1.PNG"><img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/cae_pic2.PNG">
</figure>
original images (left) and generated images (right)  

For VAE, the Reconstruction Loss, KL-Divergence Loss, and total loss are listed as follows:     
<figure class="third">
<img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_bceloss.PNG" width='250'/><img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_kldloss.PNG" width='250'/><img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_totalloss.PNG" width='250'/>
</figure>   
In addition, the video generated through linear interpolation is here:    
![image](https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_video.gif)



