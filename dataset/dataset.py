import torch.utils.data as data
import torch
# import tables
import h5py
import numpy as np
import random
from scipy import misc
from dataset.imresize import *

class DatasetFromHdf5_train(data.Dataset):
    def __init__(self, file_path, scale, patch_size):
        super(DatasetFromHdf5_train, self).__init__()

        hf = h5py.File(file_path)
        # hf = tables.open_file(file_path,driver="H5FD_CORE")

        # self.img_HR = hf.root.img_HR           # [N,ah,aw,h,w]
        # self.img_LR_2 = hf.root.img_LR_2   # [N,ah,aw,h/2,w/2]
        # self.img_LR_4 = hf.root.img_LR_4   # [N,ah,aw,h/4,w/4]

        # self.img_size = hf.root.img_size #[N,2]

        self.img_HR = hf.get('img_HR')  # [N,ah,aw,h,w]
        self.img_LR_2 = hf.get('img_LR_2')  # [N,ah,aw,h/2,w/2]
        self.img_LR_4 = hf.get('img_LR_4')  # [N,ah,aw,h/4,w/4]

        self.img_size = hf.get('img_size')  # [N,2]

        self.scale = scale
        self.psize = patch_size

    def __getitem__(self, index):

        # get one item
        hr = self.img_HR[index]  # [ah,aw,h,w]
        lr_2 = self.img_LR_2[index]  # [ah,aw,h/2,w/2]
        lr_4 = self.img_LR_4[index]  # [ah,aw,h/4,w/4]

        # crop to patch
        H, W = self.img_size[index]
        x = random.randrange(0, H - self.psize, 8)
        y = random.randrange(0, W - self.psize, 8)
        hr = hr[:, :, x:x + self.psize, y:y + self.psize]  # [ah,aw,ph,pw]
        lr_2 = lr_2[:, :, x // 2:x // 2 + self.psize // 2, y // 2:y // 2 + self.psize // 2]  # [ah,aw,ph/2,pw/2]
        lr_4 = lr_4[:, :, x // 4:x // 4 + self.psize // 4, y // 4:y // 4 + self.psize // 4]  # [ah,aw,ph/4,pw/4]

        # 4D augmentation
        # flip
        if np.random.rand(1) > 0.5:
            hr = np.flip(np.flip(hr, 0), 2)
            lr_2 = np.flip(np.flip(lr_2, 0), 2)
            lr_4 = np.flip(np.flip(lr_4, 0), 2)
            # lr_8 = np.flip(np.flip(lr_8,0),2)                
        if np.random.rand(1) > 0.5:
            hr = np.flip(np.flip(hr, 1), 3)
            lr_2 = np.flip(np.flip(lr_2, 1), 3)
            lr_4 = np.flip(np.flip(lr_4, 1), 3)
            # lr_8 = np.flip(np.flip(lr_8,1),3)
        # rotate
        r_ang = np.random.randint(1, 5)
        hr = np.rot90(hr, r_ang, (2, 3))
        hr = np.rot90(hr, r_ang, (0, 1))
        lr_2 = np.rot90(lr_2, r_ang, (2, 3))
        lr_2 = np.rot90(lr_2, r_ang, (0, 1))
        lr_4 = np.rot90(lr_4, r_ang, (2, 3))
        lr_4 = np.rot90(lr_4, r_ang, (0, 1))

        # to tensor     
        hr = hr.reshape(-1, self.psize, self.psize)  # [an,ph,pw]
        lr_2 = lr_2.reshape(-1, self.psize // 2, self.psize // 2)  # [an,phs,pws]
        lr_4 = lr_4.reshape(-1, self.psize // 4, self.psize // 4)  # [an,phs,pws]

        hr = torch.from_numpy(hr.astype(np.float32) / 255.0)
        lr_2 = torch.from_numpy(lr_2.astype(np.float32) / 255.0)
        lr_4 = torch.from_numpy(lr_4.astype(np.float32) / 255.0)

        # print(hr.shape)
        return hr, lr_2, lr_4

    def __len__(self):
        return self.img_HR.shape[0]


class DatasetFromHdf5List_train(data.Dataset):
    def __init__(self, dataroot, list_path, scale, patch_size, alpha, prob):
        super(DatasetFromHdf5List_train, self).__init__()
        self.list_path = list_path

        fd = open(self.list_path, 'r')
        self.h5files = [line.strip('\n') for line in fd.readlines()]
        print("Dataset files include {}".format(self.h5files))

        self.lens = []
        self.img_HRs = []
        self.img_LR_2s = []
        self.img_LR_4s = []
        self.img_sizes = []

        for h5file in self.h5files:
            hf = h5py.File("{}/{}".format(dataroot, h5file))
            img_HR = hf.get('img_HR')  # [N,ah,aw,h,w]
            img_LR_2 = hf.get('img_LR_2')  # [N,ah,aw,h/2,w/2]
            img_LR_4 = hf.get('img_LR_4')  # [N,ah,aw,h/4,w/4]

            img_size = hf.get('img_size')  # [N,2]

            self.lens.append(img_HR.shape[0])
            self.img_HRs.append(img_HR)
            self.img_LR_2s.append(img_LR_2)
            self.img_LR_4s.append(img_LR_4)
            self.img_sizes.append(img_size)

        self.scale = scale
        self.psize = patch_size
        self.alpha = alpha
        self.prob = prob

    def __getitem__(self, index):

        file_index = 0
        batch_index = 0
        for i in range(len(self.h5files)):
            if index < self.lens[i]:
                file_index = i
                batch_index = index
                break
            else:
                index -= self.lens[i]

        # get one item
        lfsize = self.img_sizes[file_index][batch_index, :]
        H, W = lfsize[0], lfsize[1]
        hr = self.img_HRs[file_index][batch_index, :, :, :H, :W]  # [ah,aw,h,w]
        lr_2 = self.img_LR_2s[file_index][batch_index, :, :, :H // 2, :W // 2]  # [ah,aw,h/2,w/2]
        lr_4 = self.img_LR_4s[file_index][batch_index, :, :, :H // 4, :W // 4]  # [ah,aw,h/4,w/4]

        # crop to patch
        # H, W = self.img_size[index]
        x = random.randrange(0, H - self.psize, 8)
        y = random.randrange(0, W - self.psize, 8)
        hr = hr[:, :, x:x + self.psize, y:y + self.psize]  # [ah,aw,ph,pw]
        lr_2 = lr_2[:, :, x // 2:x // 2 + self.psize // 2, y // 2:y // 2 + self.psize // 2]  # [ah,aw,ph/2,pw/2]
        lr_4 = lr_4[:, :, x // 4:x // 4 + self.psize // 4, y // 4:y // 4 + self.psize // 4]  # [ah,aw,ph/4,pw/4]

        # 4D augmentation
        # flip
        if np.random.rand(1) > 0.5:
            hr = np.flip(np.flip(hr, 0), 2)
            lr_2 = np.flip(np.flip(lr_2, 0), 2)
            lr_4 = np.flip(np.flip(lr_4, 0), 2)
            # lr_8 = np.flip(np.flip(lr_8,0),2)
        if np.random.rand(1) > 0.5:
            hr = np.flip(np.flip(hr, 1), 3)
            lr_2 = np.flip(np.flip(lr_2, 1), 3)
            lr_4 = np.flip(np.flip(lr_4, 1), 3)
            # lr_8 = np.flip(np.flip(lr_8,1),3)
        # rotate
        r_ang = np.random.randint(1, 5)
        hr = np.rot90(hr, r_ang, (2, 3))
        hr = np.rot90(hr, r_ang, (0, 1))
        lr_2 = np.rot90(lr_2, r_ang, (2, 3))
        lr_2 = np.rot90(lr_2, r_ang, (0, 1))
        lr_4 = np.rot90(lr_4, r_ang, (2, 3))
        lr_4 = np.rot90(lr_4, r_ang, (0, 1))
        # print(lr_4.shape)


        # to tensor
        hr = hr.reshape(-1, self.psize, self.psize)  # [an,ph,pw]
        lr_2 = lr_2.reshape(-1, self.psize // 2, self.psize // 2)  # [an,phs,pws]
        lr_4 = lr_4.reshape(-1, self.psize // 4, self.psize // 4)  # [an,phs,pws]

        hr_mix = hr.mean(axis=0)
        lr_2_mix = lr_2.mean(axis=0)
        lr_4_mix = lr_4.mean(axis=0)

        # print(hr_mix.shape)
        # print(lr_2_mix.shape)

        if np.random.rand(1) > self.prob:
            if self.scale==2:
                # lr_2, hr = cutblurmix(lr_2, hr, lr_2_mix, hr_mix, 2, prob=1.0, alpha=self.alpha)
                hr, lr_2 = cutmib(hr, lr_2, hr_mix, lr_2_mix, 2, prob=self.prob, alpha=self.alpha)
            else:
                # lr_4, hr = cutblurmix(lr_4, hr, lr_4_mix, hr_mix, 4, prob=1.0, alpha=self.alpha)
                hr, lr_4 = cutmib(hr, lr_4, hr_mix, lr_4_mix, 4, prob=self.prob, alpha=self.alpha)

        hr = torch.from_numpy(hr.astype(np.float32) / 255.0)
        lr_2 = torch.from_numpy(lr_2.astype(np.float32) / 255.0)
        lr_4 = torch.from_numpy(lr_4.astype(np.float32) / 255.0)

        # print(hr.shape)
        return hr, lr_2, lr_4

    def __len__(self):
        total_len = 0
        for i in range(len(self.h5files)):
            total_len += self.lens[i]

        return total_len


class DatasetFromHdf5_test(data.Dataset):
    def __init__(self, file_path, scale):
        super(DatasetFromHdf5_test, self).__init__()
        # hf = tables.open_file(file_path,driver = "H5FD_CORE")

        hf = h5py.File(file_path)
        # self.GT_y = hf["/GT_y"][0:5, :, :, :, :]  # [N,aw,ah,h,w]
        # self.LR_ycbcr = hf["/LR_ycbcr"][0:5, :, :, :, :]  # [N,ah,aw,3,h/s,w/s]
        self.GT_y = hf["/GT_y"]  # [N,aw,ah,h,w]
        self.LR_ycbcr = hf["/LR_ycbcr"]  # [N,ah,aw,3,h/s,w/s]

        # self.GT_y = hf.root.GT_y      #[N,aw,ah,h,w]
        # self.LR_ycbcr = hf.root.LR_ycbcr #[N,ah,aw,3,h/s,w/s]

        self.scale = scale

    def __getitem__(self, index):
        h = self.GT_y.shape[3]
        w = self.GT_y.shape[4]

        gt_y = self.GT_y[index]
        gt_y = gt_y.reshape(-1, h, w)
        gt_y = torch.from_numpy(gt_y.astype(np.float32) / 255.0)

        lr_ycbcr = self.LR_ycbcr[index]
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32) / 255.0)

        lr_y = lr_ycbcr[:, :, 0, :, :].clone().view(-1, h // self.scale, w // self.scale)

        lr_ycbcr_up = lr_ycbcr.view(1, -1, h // self.scale, w // self.scale)
        lr_ycbcr_up = torch.nn.functional.interpolate(lr_ycbcr_up, scale_factor=self.scale, mode='bilinear',
                                                      align_corners=False)
        lr_ycbcr_up = lr_ycbcr_up.view(-1, 3, h, w)

        return gt_y, lr_ycbcr_up, lr_y

    def __len__(self):
        return self.GT_y.shape[0]


class DatasetFromHdf5_withScale_test(data.Dataset):
    def __init__(self, file_path, scale):
        super(DatasetFromHdf5_withScale_test, self).__init__()
        # hf = tables.open_file(file_path,driver = "H5FD_CORE")

        hf = h5py.File(file_path)
        self.GT_y = hf["/GT_y"]  # [N,aw,ah,h,w]
        self.LR_ycbcr = hf["/LR_ycbcr"]  # [N,ah,aw,3,h/s,w/s]
        self.sizes = hf["/datasize"]


        self.scale = scale

    def __getitem__(self, index):

        h, w = self.sizes[index]

        gt_y = self.GT_y[index][:,:,:h,:w]
        gt_y = gt_y.reshape(-1, h, w)
        gt_y = torch.from_numpy(gt_y.astype(np.float32) / 255.0)

        lr_ycbcr = self.LR_ycbcr[index][:,:,:,:h//self.scale,:w//self.scale]
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32) / 255.0)

        lr_y = lr_ycbcr[:, :, 0, :, :].clone().view(-1, h // self.scale, w // self.scale)

        lr_ycbcr_up = lr_ycbcr.view(1, -1, h // self.scale, w // self.scale)
        lr_ycbcr_up = torch.nn.functional.interpolate(lr_ycbcr_up, scale_factor=self.scale, mode='bilinear',
                                                      align_corners=False)
        lr_ycbcr_up = lr_ycbcr_up.view(-1, 3, h, w)

        return gt_y, lr_ycbcr_up, lr_y

    def __len__(self):
        return self.GT_y.shape[0]


def cutblur(im1, im2, prob=1.0, alpha=0.7):
    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha
    # print('cut_ratio',cut_ratio)
    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    # print('ch, cw',ch, cw)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)
    # print('cy,cx', cy,cx)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
        # print(cy:cy+ch, cx:cx+cw)
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    return im1, im2


def cutmib(im1, im2, im1_mix, im2_mix, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = np.int(h_lr*cut_ratio), np.int(w_lr*cut_ratio)
    ch_hr, cw_hr = ch_lr*scale, cw_lr*scale
    cy_lr = np.random.randint(0, h_lr-ch_lr+1)
    cx_lr = np.random.randint(0, w_lr-cw_lr+1)
    cy_hr, cx_hr = cy_lr*scale, cx_lr*scale

    if np.random.random() < prob:
        if np.random.random() > 0.5:
            for i in range(an):
                im2[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = imresize(im1_mix[..., cy_hr:cy_hr+ch_hr, cx_hr:cx_hr+cw_hr], scalar_scale=1/scale)
                # im2[..., cy:cy+ch_lr, cx:cx+cw_lr] = im1[..., cy:cy+ch_lr, cx:cx+cw_lr]
            # print(cy:cy+ch_lr, cx:cx+cw_lr)
        else:
            im2_aug = im2
            for i in range(an):
                im2_aug[i] = imresize(im1[i], scalar_scale=1/scale)
                im2_aug[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = im2_mix[..., cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr]
                im2 = im2_aug

        return im1, im2
    else: 
        return im1, im2


def cutmibdifferentK(im1, im2, im1_mix, im2_mix, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im1.shape
    ch_lr, cw_lr = np.int(h_lr*cut_ratio), np.int(w_lr*cut_ratio)
    ch_hr, cw_hr = ch_lr*scale, cw_lr*scale
    cy_lr = np.random.randint(0, h_lr-ch_lr+1)
    cx_lr = np.random.randint(0, w_lr-cw_lr+1)
    cy_hr, cx_hr = cy_lr*scale, cx_lr*scale
    # apply CutBlur to inside or outside
    
    if np.random.random() < prob:
        if np.random.random() > 0.5:
            for i in range(an):
                im2[i, cy_hr:cy_hr+ch_hr, cx_hr:cx_hr+cw_hr] = imresize(im1_mix[..., cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr], scalar_scale=scale)
        else:
            im2_aug = im2
            for i in range(an):
                im2_aug[i] = imresize(im1_mix, scalar_scale=scale)
                im2_aug[i, cy_hr:cy_hr+ch_hr, cx_hr:cx_hr+cw_hr] = im2[i, cy_hr:cy_hr+ch_hr, cx_hr:cx_hr+cw_hr]
                im2 = im2_aug
        return im1, im2
    else:
        return im1, im2
