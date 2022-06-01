
import utils
import Unet_tutorial
import carvana_data
import torch
import albumentations as A
import albumentations
import torch.nn as nn
import torch.optim as optim
from Unet_tutorial import Unet
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from utils import get_loaders, check_acc, save_pred, save,save_best
import torchvision.transforms.functional as TF

import torchvision.transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
name=torch.cuda.get_device_name(device)
print(f'our cuda device is {name}')

### Hyper paramters
lrate=0.00005
epoch=2

batch=5
img_w= 500  #2000
img_h= 500  ## 2000
need_load=True
num_workers=0
pin_memory=True


path='C:/Users/sem/Documents/Ali/Unet_model/checkpoint'
best_path='Z:/Images/Ali/Maynes/best_model.pth'
image_dir='Z:/Images/Ali/Maynes/GT/'
mask_dir='Z:/Images/Ali/Maynes/GT_mask/'

val_image_dir='Z:/Images/Ali/Maynes/valid/'
val_mask_dir='Z:/Images/Ali/Maynes/valid_mask/'


# image_dir='Z:/Images/Ali/Human Liver/Nov 17/Unet/training_images/'
# mask_dir='Z:/Images/Ali/Human Liver/Nov 17/Unet/training_mask_r/'

# val_image_dir='Z:/Images/Ali/Human Liver/Nov 17/Unet/validation/'
# val_mask_dir='Z:/Images/Ali/Human Liver/Nov 17/Unet/validation_mask/'

def train_fn(loader, model, optimizer, loss_fn,scaler):
    loop=tqdm(loader)
    
    transform_normal=torchvision.transforms.Compose([
         torchvision.transforms.Normalize(142,28)                                        

       ])
    
    for batch_idx, (raw_data, target) in enumerate(loop):
        
        un_data=raw_data.float()
        target=target.unsqueeze(1).float().to(device)
        
       # data=transform_normal(un_data).unsqueeze(1).to(device)
        data=un_data.unsqueeze(1).to(device)
        
       # print(f'data max is {torch.max(data)} and target max is {torch.max(target)}')
        
        
        
        #data=data.to(device)
        #target=target.unsqueeze(1).to(device)
        #target=target.to(device)
       
        #### forward pass
        with torch.cuda.amp.autocast():
          prediction=model(data)
          loss=loss_fn(prediction,target)
       ### backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
       # scaler.update()
        
        loop.set_postfix(loss=loss.item())
        
def main():
    train_transform=A.Compose(
        [
           # A.Resize(height=img_h,width=img_w),
            A.RandomCrop(width=img_h, height=img_w),
            A.Rotate(limit=30,p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.augmentations.transforms.Downscale (scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.1),
            A.augmentations.transforms.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=False, always_apply=False, p=0.1)
    
  #         A.Normalize(
    #            mean=[217],
     #           std=[10],
      #          max_pixel_value=255,
       #         ),
        #    ToTensorV2(),
            
        
        ])

    val_transform=A.Compose(
        [
            # A.Resize(height=img_h,width=img_w),
             A.RandomCrop(width=img_h, height=img_w)
      #      A.Normalize(
       #         mean=[0],
        #        std=[1],
         #       max_pixel_value=255,
          #     ),
           # ToTensorV2(),
            
        
        ])
          

    model=Unet(in_channel=1, out_channel=1).to(device)
    loss_fn=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters(),lr=lrate)
    
    train_loader, val_loader = get_loaders(
       image_dir,
       mask_dir,
       val_image_dir,
       val_mask_dir,
       batch,
       train_transform,
       val_transform,
       num_workers=num_workers,
       pin_memory=pin_memory
   )
   
  
    scaler = torch.cuda.amp.GradScaler()

    # if need_load:
    #     load(path, model,optimizer)
        
    if need_load:
      
        status=torch.load(best_path)
        model.load_state_dict(status["state_dict"])
        optimizer.load_state_dict(status["optimizer"])
        best_score=status['score']
        print(f'loading all trained parameters with best score of {best_score}')
        model.train()
    else:
      best_score=0
    
    for items in range(epoch):
       train_fn(train_loader, model, optimizer, loss_fn,scaler)
       print(f'running epoch # {items}')
     #  check_acc(val_loader,model,device="cuda")
      
       cum_score=check_acc(val_loader,model,device="cuda")
       current_dice=cum_score/len(val_loader)
       
       if current_dice>best_score:
         best_score=current_dice
         print(f'best dice score is {current_dice:.3f}')
         checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "score": best_score,
            }
         save_best(checkpoint,best_path)
       
       save_pred(val_loader,model,folder="Z:/Images/Ali/Maynes/preds/")
       checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "score": best_score,
        }
       save(checkpoint,path)

if __name__=="__main__":
   main()      
        

