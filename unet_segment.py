# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:02:38 2022

@author: sem
"""
import utils_segment
import Unet_tutorial
import carvana_data_segment
#!pip install wandb
#!pip install albumentations==0.4.6

import torch
import albumentations as A
#import wandb
import albumentations
import torch.nn as nn
import torch.optim as optim
from Unet_tutorial import Unet
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from utils_segment import get_loaders,  save_pred
import torchvision.transforms.functional as TF
import torchvision.transforms
import statistics
import numpy
#wandb.init()
import gc


gc.collect()
torch.cuda.empty_cache()

device = "cpu"    #"cuda" if torch.cuda.is_available() else "cpu"
print(device)
#name=torch.cuda.get_device_name(device)
#print(f'our cuda device is {name}')

### Hyper paramters

epoch=1
batch=1
img_w=2000 
img_h=2000
need_load=True




best_path='Z:/Images/Ali/Maynes/best_model.pth'


val_image_dir='Z:/Images/Ali/Maynes/Input_segment/'



        
def main():
   

    val_transform=A.Compose(
        [
            A.Resize(height=img_h,width=img_w),
              
        ])
          

    model=Unet(in_channel=1, out_channel=1).to(device)
   
       
    val_loader = get_loaders(
  
       val_image_dir,       
       batch,
       val_transform,

   )
   
  
    scaler = torch.cuda.amp.GradScaler()

    if need_load:

        status=torch.load(best_path)
        model.load_state_dict(status["state_dict"])
        best_score=status['score']
        print(f'loading all trained parameters with best score of {best_score}')
        model.train()
    else:
      best_score=0
   
    
    for items in range(epoch):
       
         print(f'running epoch # {items}')                       
         save_pred(val_loader,model,folder="Z:/Images/Ali/Maynes/segmented/")


if __name__=="__main__":
   main()      
      

