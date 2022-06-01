# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 22:48:10 2021

@author: alida
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2

#image_dir=r'C:\Users\alida\.spyder-py3\image_data\Carvana\train'
#mask_dir=r'C:\Users\alida\.spyder-py3\image_data\Carvana\train_masks'

class Carvanadata(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.images=os.listdir(image_dir)
        self.masks=os.listdir(mask_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        img_path=os.path.join(self.image_dir,self.images[idx])
        mask_path=os.path.join(self.mask_dir,self.images[idx]) #.replace(".jpg","_mask.gif"))
        
        if img_path.endswith('.tiff'):
          image1=cv2.imread(img_path)
          image2=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)  
          image=np.array(image2)
       #image=np.array(Image.open(img_path))  #.convert("L"))
        
      #  scaled=(raw_image/raw_image.max())*255.
     #   image=np.uint8(scaled)
        if mask_path.endswith('.tiff'): 
          mask1=cv2.imread(mask_path)
          mask2=cv2.cvtColor(mask1,cv2.COLOR_BGR2GRAY)  
          mask=np.array(mask2)
        #  print(np.count_nonzero(mask==2 ))
          mask[mask==1]=0
          mask[mask==2]=1
       #   print(np.unique(mask))
       #   print(np.count_nonzero(mask==1 ))
      #  mask=np.array(Image.open(mask_path)) #.convert("L") , dtype=np.float32)
        ##mask[mask==2.0]=1.0
      #  mask[mask==255]=1.0
        if self.transform is not None:
          augumentation=self.transform(image=image, mask=mask)
          image=augumentation["image"]
          mask=augumentation["mask"] 
        return image, mask
    
    


