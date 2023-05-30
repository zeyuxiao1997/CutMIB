import torch.utils.data as data
import torch
# import tables
import h5py
import numpy as np
import random
from scipy import misc

class DatasetFromHdf5_train(data.Dataset):
    def __init__(self, file_path, scale, patch_size):
        super(DatasetFromHdf5_train, self).__init__()
        
        hf = h5py.File(file_path)
        # hf = tables.open_file(file_path,driver="H5FD_CORE")
        
        # self.img_HR = hf.root.img_HR           # [N,ah,aw,h,w]
        # self.img_LR_2 = hf.root.img_LR_2   # [N,ah,aw,h/2,w/2]
        # self.img_LR_4 = hf.root.img_LR_4   # [N,ah,aw,h/4,w/4]
        
        # self.img_size = hf.root.img_size #[N,2]

        self.img_HR = hf.get('img_HR')           # [N,ah,aw,h,w]
        self.img_LR_2 = hf.get('img_LR_2')   # [N,ah,aw,h/2,w/2]
        self.img_LR_4 = hf.get('img_LR_4')   # [N,ah,aw,h/4,w/4]
        
        self.img_size = hf.get('img_size') #[N,2]
        
        self.scale = scale        
        self.psize = patch_size
    
    def __getitem__(self, index):
                        
        # get one item
        hr = self.img_HR[index]       # [ah,aw,h,w]
        lr_2 = self.img_LR_2[index]   # [ah,aw,h/2,w/2]
        lr_4 = self.img_LR_4[index]   # [ah,aw,h/4,w/4]
                                               
        # crop to patch
        H, W = self.img_size[index]
        x = random.randrange(0, H-self.psize, 8)    
        y = random.randrange(0, W-self.psize, 8) 
        hr = hr[:, :, x:x+self.psize, y:y+self.psize] # [ah,aw,ph,pw]
        lr_2 = lr_2[:, :, x//2:x//2+self.psize//2, y//2:y//2+self.psize//2] # [ah,aw,ph/2,pw/2]
        lr_4 = lr_4[:, :, x//4:x//4+self.psize//4, y//4:y//4+self.psize//4] # [ah,aw,ph/4,pw/4]  

        # 4D augmentation
        # flip
        if np.random.rand(1)>0.5:
            hr = np.flip(np.flip(hr,0),2)
            lr_2 = np.flip(np.flip(lr_2,0),2)
            lr_4 = np.flip(np.flip(lr_4,0),2)  
            # lr_8 = np.flip(np.flip(lr_8,0),2)                
        if np.random.rand(1)>0.5:
            hr = np.flip(np.flip(hr,1),3)
            lr_2 = np.flip(np.flip(lr_2,1),3)
            lr_4 = np.flip(np.flip(lr_4,1),3) 
            # lr_8 = np.flip(np.flip(lr_8,1),3)
        # rotate
        r_ang = np.random.randint(1,5)
        hr = np.rot90(hr,r_ang,(2,3))
        hr = np.rot90(hr,r_ang,(0,1))
        lr_2 = np.rot90(lr_2,r_ang,(2,3))
        lr_2 = np.rot90(lr_2,r_ang,(0,1))           
        lr_4 = np.rot90(lr_4,r_ang,(2,3))
        lr_4 = np.rot90(lr_4,r_ang,(0,1)) 

        # to tensor     
        hr = hr.reshape(-1,self.psize,self.psize) # [an,ph,pw]
        lr_2 = lr_2.reshape(-1,self.psize//2,self.psize//2) #[an,phs,pws]
        lr_4 = lr_4.reshape(-1,self.psize//4,self.psize//4) # [an,phs,pws]

        hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        lr_2 = torch.from_numpy(lr_2.astype(np.float32)/255.0)  
        lr_4 = torch.from_numpy(lr_4.astype(np.float32)/255.0)

        # print(hr.shape)
        if self.scale==2:
            return hr, lr_2
        elif self.scale==4:
            return hr, lr_4

    def __len__(self):
        return self.img_HR.shape[0]

class DatasetFromHdf5_test(data.Dataset):
    def __init__(self, file_path, scale):
        super(DatasetFromHdf5_test, self).__init__()
        # hf = tables.open_file(file_path,driver = "H5FD_CORE")
        
        hf = h5py.File(file_path)       
        self.GT_y = hf["/GT_y"][0:5, :, :, :, :]  #[N,aw,ah,h,w]
        self.LR_ycbcr = hf["/LR_ycbcr"][0:5, :, :, :, :]  #[N,ah,aw,3,h/s,w/s]

        # self.GT_y = hf.root.GT_y      #[N,aw,ah,h,w]
        # self.LR_ycbcr = hf.root.LR_ycbcr #[N,ah,aw,3,h/s,w/s]

        self.scale = scale

    def __getitem__(self, index):

        h = self.GT_y.shape[3]
        w = self.GT_y.shape[4]
        
        gt_y = self.GT_y[index]
        gt_y = gt_y.reshape(-1, h, w)
        gt_y = torch.from_numpy(gt_y.astype(np.float32)/255.0)

        lr_ycbcr = self.LR_ycbcr[index]
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32)/255.0)       

        lr_y = lr_ycbcr[:, :, 0, :, :].clone().view(-1, h//self.scale, w//self.scale)
        
        lr_ycbcr_up = lr_ycbcr.view(1, -1, h//self.scale, w//self.scale)
        lr_ycbcr_up = torch.nn.functional.interpolate(lr_ycbcr_up, scale_factor=self.scale, mode='bilinear',align_corners=False)
        lr_ycbcr_up = lr_ycbcr_up.view(-1, 3, h, w)
        
        return gt_y, lr_ycbcr_up, lr_y 
        
    def __len__(self):
        return self.GT_y.shape[0]