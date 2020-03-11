## Project Title

Generative Models: Convolutional Autoencoders and Variational Autoencoders

### Task and Model

The task is to generate as similar images (fake images) as the given dataset (real images).  
The Convolutional AutoEncoder(CAE) and Variational AutoEncoder(VAE) are created here, to reconstruct images for the UT Zappos50K Dataset.


### Prerequisites

I uploaded the zipped dataset(625 Mb) to Google drive, and used Google Colab to load & unzip the data, the dataset can be found here https://drive.google.com/file/d/1nYEgytPOkFyUjDQfBGzwCQbszf6OE143/view?usp=sharing. You can also directly upload the unzipped dataset to your Colab, just remember to change the path in Step1 -'DATASETS/UTZappos50K'.


### Introduction

The UT Zappos50K Dataset used here contains 4 types of images, i.e. Boots, Sandals, Shoes, Slippers. The Adam optimizer was utilized with learning rate of 0.001. As for the loss function, for CAE, I used mean squared loss; For VAE, I used binary cross entropy (reconstruction loss) plus KL-divergence (regularization loss).  
<div align=center><img src="http://chart.googleapis.com/chart?cht=tx&chl= $$ L_{reconstruciton} = \minus\frac{1}{n} \sum_{i}^{n}(x_i log(f(z_i)) \plus (1\minus x_i) log(1\minus f(z_i))) $$" style="border:none;"></div>     
<div align=center><img src="http://chart.googleapis.com/chart?cht=tx&chl= $$ L_{regularization} = \frac{1}{2n}\sum_{i}^{n}(\mu_{i}^{2} \p \sigma_{i}^2 \m log(\sigma_i^2)\m 1)$$  " style="border:none;"></div>    
<div align=center><img src="http://chart.googleapis.com/chart?cht=tx&chl= $$ L_{loss} =L_{regularization} \p L_{reconstruction}$$ " style="border:none;"></div>   
 
### Data Visualization
#### CAE outcome display
For CAE, the training loss looks like:   
<div align=center><img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/cae_loss.PNG" width='320'/></div>    

the performance of trained CAE model can be seen below:    
<div align=center><img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/cae_pic2.PNG" width='300'/></div>    
<div align=center>original images(left) v.s generated images(right)</div>       

#### VAE outcome display
For VAE, the Reconstruction Loss, KL-Divergence Loss, and total loss are listed as follows:   
<div align=center><figure class="third">
<img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_bceloss.PNG" width='270'/><img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_kldloss.PNG" width='270'/><img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_totalloss.PNG" width='270'/>
</figure></div>  

the performance of trained VAE model is shown below:
<div align=center><figure class="half">
<img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_orig_img.PNG" width='270'/><img src="https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_gen_img.PNG" width='270'/>
</figure></div> 
<div align=center>original images(left) v.s generated images(right)</div>  

Last but not least, the video obtained from linear interpolation:   
<div align=center><img src='https://github.com/MianWang123/Variational-Autoencoder/blob/master/pics/vae_video.gif'></div>
