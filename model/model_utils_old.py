# @Time = 2021.12.23
# @Author = Zhen

"""
Model utilities.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from utils.convNd import convNd

class SAS_para:
    def __init__(self):
        self.act = 'lrelu'
        self.fn = 64

class SAC_para:
    def __init__(self):
        self.act = 'lrelu'
        self.symmetry = True
        self.max_k_size = 3
        self.fn = 64

class SAS_conv(nn.Module):
    def __init__(self, act='relu', fn=64):
        super(SAS_conv, self).__init__()

        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        self.spaconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.spaconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.spaconv.bias, 0.0)

        self.angconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.angconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.angconv.bias, 0.0)

    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(N * U * V, c, h, w)

        out = self.act(self.spaconv(x))  # [N*U*V,c,h,w]
        out = out.view(N, U * V, c, h * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * h * w, c, U, V)  # [N*h*w,c,U,V]

        out = self.act(self.angconv(out))  # [N*h*w,c,U,V]
        out = out.view(N, h * w, c, U * V)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, U, V, c, h, w)  # [N,U,V,c,h,w]
        out = out.permute(0, 3, 1, 2, 4, 5).contiguous()  # [N,c,U,V,h,w]
        return out


class SAC_conv(nn.Module):
    def __init__(self, act='relu', symmetry=True, max_k_size=3, fn=64):
        super(SAC_conv, self).__init__()

        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        if symmetry:
            k_size_ang = max_k_size
            k_size_spa = max_k_size
        else:
            k_size_ang = max_k_size - 2
            k_size_spa = max_k_size

        self.verconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1, 1), padding=(k_size_ang // 2, k_size_spa // 2))
        init.kaiming_normal_(self.verconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.verconv.bias, 0.0)

        self.horconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1, 1), padding=(k_size_ang // 2, k_size_spa // 2))
        init.kaiming_normal_(self.horconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.horconv.bias, 0.0)

    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N * V * w, c, U, h)

        out = self.act(self.verconv(x))  # [N*V*w,c,U,h]
        out = out.view(N, V * w, c, U * h)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * U * h, c, V, w)  # [N*U*h,c,V,w]

        out = self.act(self.horconv(out))  # [N*U*h,c,V,w]
        out = out.view(N, U * h, c, V * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, V, w, c, U, h)  # [N,V,w,c,U,h]
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()  # [N,c,U,V,h,w]
        return out


class SAC_conv_for_test(nn.Module):
    def __init__(self, act='relu', symmetry=True, max_k_size=3, fn=64):
        super(SAC_conv_for_test, self).__init__()

        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        if symmetry:
            k_size_ang = max_k_size
            k_size_spa = max_k_size
        else:
            k_size_ang = max_k_size - 2
            k_size_spa = max_k_size

        self.verconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1, 1), padding=(k_size_ang // 2, k_size_spa // 2))
        init.kaiming_normal_(self.verconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.verconv.bias, 0.0)

        self.horconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1, 1), padding=(k_size_ang // 2, k_size_spa // 2))
        init.kaiming_normal_(self.horconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.horconv.bias, 0.0)

    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N * V * w, c, U, h)

        out = self.act(self.verconv(x))  # [N*V*w,c,U,h]
        out = out.view(N, V * w, c, U * h)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, U, h, c, V, w)  # [N,U,h,c,V,w]

        out_horconv = []
        for u in range(U):
            out_slice = out[:, u, :, :, :, :]  # [N, h, c, V, w]
            out_slice = out_slice.contiguous().view(N * h, c, V, w)
            out_horconv_slice = self.act(self.horconv(out_slice))  # [N*h, c, V, w]
            out_horconv_slice = out_horconv_slice.view(N, 1, h, c, V, w)  # [N, 1, h, c, V, w]
            out_horconv.append(out_horconv_slice)
        out_horconv = torch.cat(out_horconv, dim=1)  # [N, U, h, c, V, w]
        out_horconv = out_horconv.permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, c, U, V, h, w]
        return out_horconv


class SAV_concat(nn.Module):
    def __init__(self, SAS_para, SAC_para, residual_connection=True):
        """
        parameters for building SAS-SAC block
        :param SAS_para: {act, fn}
        :param SAC_para: {act, symmetry, max_k_size, fn}
        :param residual_connection: True or False for residual connection
        """
        super(SAV_concat, self).__init__()
        self.res_connect = residual_connection
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size, fn=SAC_para.fn)

    def forward(self, lf_input):
        feat = self.SAS_conv(lf_input)
        res = self.SAC_conv(feat)
        if self.res_connect:
            res += lf_input
        return res


class SAV_parallel(nn.Module):
    def __init__(self, SAS_para, SAC_para, feature_concat=True):
        super(SAV_parallel, self).__init__()
        self.feature_concat = feature_concat
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size, fn=SAC_para.fn)
        self.lrelu_para = nn.LeakyReLU(negative_slope=0.2, inplace=True)  #
        if self.feature_concat:
            self.channel_reduce = convNd(in_channels=2 * SAS_para.fn,
                                         out_channels=SAS_para.fn,
                                         num_dims=4,
                                         kernel_size=(1, 1, 1, 1),
                                         stride=(1, 1, 1, 1),
                                         padding=(0, 0, 0, 0),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                              'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        # for m in self.modules():
        #     weights_init_kaiming_small(m, 'relu', a=0.2, scale=0.1)

    def forward(self, lf_input):
        sas_feat = self.SAS_conv(lf_input)
        # sas_feat = self.lrelu_para(sas_feat)#
        sac_feat = self.SAC_conv(lf_input)  # [N,c,U,V,h,w]
        # sac_feat = self.lrelu_para(sac_feat)

        if self.feature_concat:
            concat_feat = torch.cat((sas_feat, sac_feat), dim=1)  # [N,2c,U,V,h,w]
            feat = self.lrelu_para(concat_feat)
            res = self.channel_reduce(feat)
            res += lf_input
        else:
            res = sas_feat + sac_feat + lf_input
        return res


class SAV_concat_for_test(nn.Module):
    def __init__(self, SAS_para, SAC_para, residual_connection=True):
        """
        parameters for building SAS-SAC block
        :param SAS_para: {act}
        :param SAC_para: {act, symmetry, max_k_size}
        :param residual_connection: True or False for residual connection
        """
        super(SAV_concat_for_test, self).__init__()
        self.res_connect = residual_connection
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv_for_test(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size, fn=SAC_para.fn)

    def forward(self, lf_input):
        feat = self.SAS_conv(lf_input)
        res = self.SAC_conv(feat)
        if self.res_connect:
            res += lf_input
        return res


class SAV_parallel_for_test(nn.Module):
    def __init__(self, SAS_para, SAC_para, feature_concat=True):
        super(SAV_parallel_for_test, self).__init__()
        self.feature_concat = feature_concat
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv_for_test(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size, fn=SAC_para.fn)
        self.lrelu_para = nn.LeakyReLU(negative_slope=0.2, inplace=True)  #
        if self.feature_concat:
            self.channel_reduce = convNd(in_channels=2 * SAS_para.fn,
                                         out_channels=SAS_para.fn,
                                         num_dims=4,
                                         kernel_size=(1, 1, 1, 1),
                                         stride=(1, 1, 1, 1),
                                         padding=(0, 0, 0, 0),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                              'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        # for m in self.modules():
        #     weights_init_kaiming_small(m, 'relu', a=0.2, scale=0.1)

    def forward(self, lf_input):
        sas_feat = self.SAS_conv(lf_input)
        # sas_feat = self.lrelu_para_test(sas_feat)#
        sac_feat = self.SAC_conv(lf_input)  # [N,c,U,V,h,w]
        # sas_feat = self.lrelu_para_test(sas_feat)

        if self.feature_concat:
            concat_feat = torch.cat((sas_feat, sac_feat), dim=1)  # [N,2c,U,V,h,w]
            feat = self.lrelu_para(concat_feat)
            res = self.channel_reduce(feat)
            res += lf_input
        else:
            # concat_feat = sas_feat + sac_feat
            # feat = self.lrelu_para(concat_feat)
            # res = feat + lf_input
            concat_feat = sas_feat + sac_feat
            feat = self.lrelu_para(concat_feat)
            res = feat + lf_input
            res = self.lrelu_para(res)
        return res


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_kaiming_small(m, act, a, scale):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose3d') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_kaiming_for_all_small(m, act, scale, a=0):
    # if act="relu", a use the default value
    # if act="leakyrelu", a use the negative slope of leakyrelu
    nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
    m.weight.data *= scale
    if m.bias is not None:
        m.bias.data.zero_()


def weights_init_kaiming_for_all(m, act, a=0):
    nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
    if m.bias is not None:
        m.bias.data.zero_()


def getNetworkDescription(network):
    # if isinstance(network, nn.DataParallel) or isinstance(network, nn.DistributedDataParallel):
    if isinstance(network, nn.DataParallel):
        network = network.module
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n