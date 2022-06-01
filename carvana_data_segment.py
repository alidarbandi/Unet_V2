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
    def __init__(self, image_dir): #,  transform):
        self.image_dir=image_dir
        
   #     self.transform=transform
        self.images=os.listdir(image_dir)
      
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        img_path=os.path.join(self.image_dir,self.images[idx])
        
        
        #if img_path.endswith('.tiff'):
        image1=cv2.imread(img_path)
        image2=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)  
        image=np.array(image2)
       #image=np.array(Image.open(img_path))  #.convert("L"))
        
    
      #  if self.transform is not None:
      #      augumentation=self.transform(image=image)
       #     image=augumentation["image"]
          
        return image
    
    


