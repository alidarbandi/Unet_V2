
import torch
import torchvision
from carvana_data_segment import Carvanadata
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
def save(status,path):
  torch.save(status, os.path.join(path,'model.pth'))
  print("saving the model and optimizer parameters ")
  
  
def save_best(best_status,best_path):
  torch.save(best_status, best_path)
  print("saving the BEST model ")  
  

  
def get_loaders(
    val_image_dir,
    batch,
    val_transform,
   # num_workers=4,
   # pin_memory=True,
):
    

    val_ds = Carvanadata(
        image_dir=val_image_dir,
   #     transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
#        num_workers=num_workers,
#        pin_memory=pin_memory,
    )

    return  val_loader



  
  

def save_pred(loader, model, folder="Z:/Images/Ali/Maynes/segmented/"
 ,device="cpu"):
  
  with torch.no_grad():
   model.eval()
  for idx, x in enumerate(loader):
       x=x.to(device).unsqueeze(1).float()
       with torch.no_grad():
        preds=torch.sigmoid(model(x))
        preds=(preds>0.5).float()
        
        
       for cc in range(0,1):     ##range(0,5)
        
      #   imx=x[cc,:,:].byte().cpu().numpy().squeeze(axis=0)
         #imx=cv2.cvtColor(imx, cv2.COLOR_BGR2GRAY)
         
         
         imp=preds[cc,:,:].byte().cpu().numpy().squeeze(axis=0)
         
      ##    cv2.imwrite(folder+f"y_{idx}{cc}.tif",imy)
         cv2.imwrite(folder+f"pred_{idx}{cc}.tif",imp)
       
  model.train()
































