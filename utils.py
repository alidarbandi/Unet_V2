
import torch
import torchvision
from carvana_data import Carvanadata
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
    image_dir,
    mask_dir,
    val_image_dir,
    val_mask_dir,
    batch,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = Carvanadata(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_ds = Carvanadata(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader , val_loader

def check_acc(loader,model,device="cuda"):
   num_correct=0
   num_pixels=0
   dice=0
   model.eval()
   with torch.no_grad():
     for x,y in loader:
       x=x.to(device).unsqueeze(1).float()
       y=y.to(device).unsqueeze(1).to("cpu")
       preds=torch.sigmoid(model(x))
       preds=(preds>0.5).float().to("cpu")
       
       #print(f"shape y is {y.size()} and x is {x.size()} and preds is {preds.size()}")
       preds=torch.flatten(preds)
       y=torch.flatten(y)
   #    print(f"shape y is {y.size()}, and preds is {preds.size()}")
   #    print(f"max preds is {torch.max(preds)}")
   #    print(f"max y is {torch.max(y)}")
       preds=np.array(preds)
       y=np.array(y)
       
       num_correct+=(preds==y).sum()
       #num_pixels+=torch.numel(preds)
       num_pixels+=preds.size
    #   print (f'number of correct is {num_correct} and total pixel is {num_pixels}')
       
       dice+=(2*((preds*y).sum()))/((preds + y).sum())
       
   accp=(num_correct/num_pixels)*100
   
       
  # print(f' validation accuracy of {accp}')
  # print(f'Dice score is {dice/len(loader):.2f} ')

   model.train()
   return dice

def save_pred(loader, model, folder="Z:/Images/Ali/Maynes/preds/"
 ,device="cuda"):
  
  model.eval()
  for idx, (x,y) in enumerate(loader):
       x=x.to(device).unsqueeze(1).float()
       with torch.no_grad():
        preds=torch.sigmoid(model(x))
        preds=(preds>0.5).float()
        yy=y.unsqueeze(1)
       for cc in range(0,2):     ##range(0,5)
        # print(f'size of x loader is {x.size()}')
         imx=x[cc,:,:].byte().cpu().numpy().squeeze(axis=0)
         #imx=cv2.cvtColor(imx, cv2.COLOR_BGR2GRAY)
         
         imy=yy[cc,:,:].byte().cpu().numpy().squeeze(axis=0)
         imp=preds[cc,:,:].byte().cpu().numpy().squeeze(axis=0)
         
         cv2.imwrite(folder+f"x_{idx}{cc}.tif",imx)
         cv2.imwrite(folder+f"y_{idx}{cc}.tif",imy)
         cv2.imwrite(folder+f"pred_{idx}{cc}.tif",imp)
        # torchvision.utils.save_image(imp, f"{folder}pred_{idx}{cc}.png")
       #  torchvision.utils.save_image(imy, f"{folder}y_{idx}{cc}.png")
        # torchvision.utils.save_image(imx, f"{folder}x_{idx}{cc}.png")
  model.train()
































