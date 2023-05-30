import torch.utils.data as data
import torch
import h5py
import numpy as np
import random

class DatasetFromHdf5List_train(data.Dataset):
    def __init__(self, dataroot, list_path, scale, patch_size):
        super(DatasetFromHdf5List_train, self).__init__()

        # hf = tables.open_file(file_path,driver="H5FD_CORE")

        # self.img_HR = hf.root.img_HR           # [N,ah,aw,h,w]
        # self.img_LR_2 = hf.root.img_LR_2   # [N,ah,aw,h/2,w/2]
        # self.img_LR_4 = hf.root.img_LR_4   # [N,ah,aw,h/4,w/4]

        # self.img_size = hf.root.img_size #[N,2]
        self.list_path = list_path
        self.batchsize = 1

        fd = open(self.list_path, 'r')
        self.h5files = [line.strip('\n') for line in fd.readlines()]
        print("Dataset files include {}".format(self.h5files))

        self.lens = []
        self.img_HRs = []
        self.img_LR_2s = []
        self.img_LR_4s = []
        self.img_sizes = []

        for i in range(self.batchsize):
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

        self.an = 5
        self.scale = scale
        self.psize = patch_size

    def __getitem__(self, index):

        file_index = 0
        batch_index = 0
        for i in range(self.batchsize*len(self.h5files)):
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

        # to tensor
        hr = hr.reshape(-1, self.psize, self.psize)  # [an,ph,pw]
        lr_2 = lr_2.reshape(-1, self.psize // 2, self.psize // 2)  # [an,phs,pws]
        lr_4 = lr_4.reshape(-1, self.psize // 4, self.psize // 4)  # [an,phs,pws]

        hr = torch.from_numpy(hr.astype(np.float32) / 255.0)
        lr_2 = torch.from_numpy(lr_2.astype(np.float32) / 255.0)
        lr_4 = torch.from_numpy(lr_4.astype(np.float32) / 255.0)

        # print(hr.shape)
        # return hr, lr_2, lr_4
        if self.scale==2:
            return hr, lr_2
        elif self.scale==4:
            return hr, lr_4

    def __len__(self):
        total_len = 0
        for i in range(self.batchsize*len(self.h5files)):
            total_len += self.lens[i]

        return total_len

class DatasetFromHdf5_test(data.Dataset):
    def __init__(self, file_path, scale):
        super(DatasetFromHdf5_test, self).__init__()

        hf = h5py.File(file_path)
        self.GT_y = hf["/GT_y"]  # [N,aw,ah,h,w]
        self.LR_ycbcr = hf["/LR_ycbcr"] # [N,ah,aw,3,h/s,w/s]
        self.datasize = hf["/datasize"]

        self.scale = scale

    def __getitem__(self, index):

        # get one item
        lfsize = self.datasize[index, :]
        H, W = lfsize[0], lfsize[1]

        gt_y = self.GT_y[index, :, :, :H, :W]
        gt_y = gt_y.reshape(-1, H, W)
        gt_y = torch.from_numpy(gt_y.astype(np.float32) / 255.0)

        lr_ycbcr = self.LR_ycbcr[index, :, :, :, :H // self.scale, :W // self.scale]
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32) / 255.0)

        lr_y = lr_ycbcr[:, :, 0, :, :].clone().view(-1, H // self.scale, W // self.scale)

        lr_ycbcr_up = lr_ycbcr.view(1, -1, H // self.scale, W // self.scale)
        lr_ycbcr_up = torch.nn.functional.interpolate(lr_ycbcr_up, scale_factor=self.scale, mode='bilinear',
                                                      align_corners=False)
        lr_ycbcr_up = lr_ycbcr_up.view(-1, 3, H, W)

        return gt_y, lr_ycbcr_up, lr_y

    def __len__(self):
        return self.GT_y.shape[0]


class DatasetFromHdf5List_train_add(data.Dataset):
    def __init__(self, dataroot, list_path, scale, patch_size, addition_path):
        super(DatasetFromHdf5List_train_add, self).__init__()

        # hf = tables.open_file(file_path,driver="H5FD_CORE")

        # self.img_HR = hf.root.img_HR           # [N,ah,aw,h,w]
        # self.img_LR_2 = hf.root.img_LR_2   # [N,ah,aw,h/2,w/2]
        # self.img_LR_4 = hf.root.img_LR_4   # [N,ah,aw,h/4,w/4]

        # self.img_size = hf.root.img_size #[N,2]
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

        hf = h5py.File(addition_path)
        img_HR = hf['img_HR'][:, 1:6, 1:6]   # [N,ah,aw,h,w]
        img_LR_2 = hf['img_LR_2'][:, 1:6, 1:6]   # [N,ah,aw,h/2,w/2]
        img_LR_4 = hf['img_LR_4'][:, 1:6, 1:6]   # [N,ah,aw,h/4,w/4]
        img_size = hf['img_size']
        self.lens.append(img_HR.shape[0])
        self.img_HRs.append(img_HR)
        self.img_LR_2s.append(img_LR_2)
        self.img_LR_4s.append(img_LR_4)
        self.img_sizes.append(img_size)

        self.an = 5
        self.scale = scale
        self.psize = patch_size

    def __getitem__(self, index):

        file_index = 0
        batch_index = 0
        for i in range(1+len(self.h5files)):
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

        # to tensor
        hr = hr.reshape(-1, self.psize, self.psize)  # [an,ph,pw]
        lr_2 = lr_2.reshape(-1, self.psize // 2, self.psize // 2)  # [an,phs,pws]
        lr_4 = lr_4.reshape(-1, self.psize // 4, self.psize // 4)  # [an,phs,pws]

        hr = torch.from_numpy(hr.astype(np.float32) / 255.0)
        lr_2 = torch.from_numpy(lr_2.astype(np.float32) / 255.0)
        lr_4 = torch.from_numpy(lr_4.astype(np.float32) / 255.0)

        # print(hr.shape)
        # return hr, lr_2, lr_4
        if self.scale==2:
            return hr, lr_2
        elif self.scale==4:
            return hr, lr_4

    def __len__(self):
        total_len = 0
        for i in range(1+len(self.h5files)):
            total_len += self.lens[i]
        return total_len
