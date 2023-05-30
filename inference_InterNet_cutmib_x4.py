import time
import argparse
import torch.backends.cudnn as cudnn
from dataset_dual.lfdata_utils import *
import importlib
import numpy as np
import os
import torch
from utils.logger import make_logs

##################################################
from model.LF_InterNet_wang import Net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--angRes_in", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument("--scale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument("--scale", type=int, default=4, help="SR factor")
    parser.add_argument("--angular_num", type=int, default=5, help="SR factor")
    parser.add_argument('--testset_dir', type=str, default='/gdata1/xiaozy/BoostLFSR/Wangdata/TestData_4xSR_5x5/')

    parser.add_argument("--patchsize", type=int, default=96, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=64, help="The stride between two test patches is set to patchsize/2")

    parser.add_argument('--model_path', type=str, default='/gdata1/xiaozy/BoostLFSR/Final_inference/CutMIB_model/InterNet_Bicubic_x4.pth')
    parser.add_argument('--save_path', type=str, default='./Results/')
    parser.add_argument('--MGPU', type=int, default=0)

    return parser.parse_args()


def init(cfg, test_Names, test_loaders):

    # MODEL = importlib.import_module(MODEL_PATH)
    net = Net(cfg)

    if cfg.MGPU:
        net = torch.nn.DataParallel(net, device_ids=[0])
    net.cuda()
    cudnn.benchmark = True


    make_logs("{}log/{}/".format('/gdata1/xiaozy/BoostLFSR/Final_inference/CutMIB_result', 'InterNet_x4'), "train_log.log", "train_err.log")

    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
        # net.load_state_dict(model['state_dict'])
        net.load_state_dict(model['model'])
        #             checkpoint = torch.load(resume_path)
        # model.load_state_dict(checkpoint['model'])
    else:
        print("=> no model found")

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test = inference(test_loader, test_name, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (
            test_name, psnr_epoch_test, ssim_epoch_test))
            # print(outLF.shape)
            pass
        pass



def inference(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().cuda()  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angRes, vw // cfg.angRes
        subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.stride)  # numU, numV, h*angRes, w*angRes
        numU, numV, H, W = subLFin.shape
        subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize * cfg.upscale_factor, cfg.angRes * cfg.patchsize * cfg.upscale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.cuda())
                    subLFout[u, v, :, :] = out.squeeze()

        outLF = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.upscale_factor, cfg.stride * cfg.upscale_factor, h0 * cfg.upscale_factor, w0 * cfg.upscale_factor)

        psnr, ssim = cal_metrics(label, outLF, cfg.angRes)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())


    return outLF, psnr_epoch_test, ssim_epoch_test


def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    init(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
