import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from sync_batchnorm import SynchronizedBatchNorm2d
from miscc.config import cfg
BatchNorm = SynchronizedBatchNorm2d
# BatchNorm = torch.nn.BatchNorm2d
InstanceNorm = torch.nn.InstanceNorm2d

def conv1x1(in_channel, out_channel):
    "1x1 convolution with padding"
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

def conv3x3(in_channel, out_channel):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

def conv4x4(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False)

def get_image(in_channel, out_channel=3):
    block = nn.Sequential(
        BatchNorm(in_channel),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
        nn.Tanh()
    )
    return block

def get_mask(in_channel, mask_channel=100, out_channel=1):
    block = nn.Sequential(
        conv3x3(in_channel, mask_channel),
        BatchNorm(mask_channel),
        nn.ReLU(),
        conv1x1(mask_channel, out_channel)
    )
    return block

class affine_ssa(nn.Module):

    # the function refers to https://github.com/wtliao/text2image
    def __init__(self, num_features, ntf=cfg.TEXT.EMBEDDING_DIM, norm_layer=BatchNorm):
        super(affine_ssa, self).__init__()

        # nn.InstanceNorm2d whose affine is set as False in default
        self.norm2d = norm_layer(num_features, affine=False)
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(ntf, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(ntf, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.zeros_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, feat, cond, semi_mask):

        feat = self.norm2d(feat)
        weight = self.fc_gamma(cond)
        bias = self.fc_beta(cond)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = feat.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        weight = weight * semi_mask + 1
        bias = bias * semi_mask
        return weight * feat + bias

# Spatial Affine Generative Blocks (SAGB)
class SAGB_Block(nn.Module):

    def __init__(self, in_ch, out_ch, affine_blocks=None, pred_mask=True):
        super(SAGB_Block, self).__init__()
        if affine_blocks is None:
            affine_blocks = [affine_ssa, affine_ssa]
        self.learnable_sc = in_ch != out_ch
        self.pred_mask = pred_mask
        self.c1 = conv3x3(in_ch, out_ch)
        self.c2 = conv3x3(out_ch, out_ch)
        self.affine1 = affine_blocks[0](in_ch)
        self.affine2 = affine_blocks[1](out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

        if self.pred_mask:
            self.conv_mask = get_mask(out_ch)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, feat, conds, semi_mask):
        c_feat = self.affine1(feat, conds[0], semi_mask)
        c_feat = nn.ReLU(inplace=True)(c_feat)
        c_feat = self.c1(c_feat)
        c_feat = self.affine2(c_feat, conds[1], semi_mask)
        c_feat = nn.ReLU(inplace=True)(c_feat)
        return self.c2(c_feat)

    def forward(self, feat, conds, semi_mask):
        c_feat = self.shortcut(feat) + self.gamma * self.residual(feat, conds, semi_mask)
        c_semi_mask = None
        if self.pred_mask:
            c_semi_mask = self.conv_mask(c_feat)
        return c_feat, c_semi_mask

# Cumulative Blocks
class Cum_Block(nn.Module):
    def __init__(self, prev_channel, cur_channel):
        super(Cum_Block, self).__init__()
        self.up_block = nn.Sequential(
            conv1x1(prev_channel, cur_channel),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(cur_channel, cur_channel)
        )
        self.fuse_block = conv3x3(cur_channel, cur_channel)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, prev_feat, cur_feat):
        prev_feat = self.up_block(prev_feat)
        fused_feat = self.fuse_block(prev_feat + cur_feat * self.gamma)
        return fused_feat

# Attribute Enhancing
class ATTR_Enhance(nn.Module):

    def __init__(self, ntf=cfg.TEXT.EMBEDDING_DIM):
        super().__init__()
        self.attr_query = nn.Linear(ntf, ntf)
        self.attr_key = nn.Linear(ntf, ntf)
        self.attr_value = nn.Linear(ntf, ntf)
        self._norm_fact = 1 / math.sqrt(ntf)

    def forward(self, sent, attrs):
        """
        sent: bs x ntf
        attrs: bs x attr_num x ntf
        basic data_flow:
        """
        sent = sent.unsqueeze(1)
        combine = torch.cat([sent, attrs], dim=1)
        q = self.attr_query(combine)
        k = self.attr_key(combine)
        v = self.attr_value(combine)
        attn_attrs = nn.Softmax(dim=-1)(torch.bmm(q, k.permute(0, 2, 1))) * self._norm_fact
        attn_attrs = torch.bmm(attn_attrs, v)
        attn_sent = attn_attrs[:,0,:]
        return attn_sent, attn_attrs

    @staticmethod
    def attr_merge(attn_attrs):
        # method 1.
        res_attr = attn_attrs.sum(dim=1)
        # method 2.
        # attn_attrs.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        # attn_attrs.sum(dim=1, keepdim=True)
        # # bs x 1 x ntf
        # res_attr = torch.log(attn_attrs).squeeze()
        return res_attr


class Gen(nn.Module):
    def __init__(self, ngf=cfg.GAN.GF_DIM, nz=cfg.GAN.Z_DIM):
        super(Gen, self).__init__()
        self.ngf = ngf

        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        self.blocks = nn.ModuleList([
            SAGB_Block(ngf * 8, ngf * 8, [affine_ssa, affine_ssa], pred_mask=True),
            # 4 x 4
            SAGB_Block(ngf * 8, ngf * 8, [affine_ssa, affine_ssa], pred_mask=True),
            # 8 x 8
            SAGB_Block(ngf * 8, ngf * 8, [affine_ssa, affine_ssa], pred_mask=True),
            # 16 x 16
            SAGB_Block(ngf * 8, ngf * 8, [affine_ssa, affine_ssa], pred_mask=True),
            # 32 x 32
            SAGB_Block(ngf * 8, ngf * 4, [affine_ssa, affine_ssa], pred_mask=True),
            # 64 x 64
            SAGB_Block(ngf * 4, ngf * 2, [affine_ssa, affine_ssa], pred_mask=True),
            # 128 x 128
            SAGB_Block(ngf * 2, ngf * 1, [affine_ssa, affine_ssa], pred_mask=False)
            # 256 x 256
        ])

        self.cum_64 = Cum_Block(ngf * 8, ngf * 4)
        self.cum_128 = Cum_Block(ngf * 4, ngf * 2)
        self.cum_256 = Cum_Block(ngf * 2, ngf * 1)

        self.get_image_64 = get_image(ngf*4, 3)
        self.get_image_128 = get_image(ngf*2, 3)
        self.get_image_256 = get_image(ngf, 3)

        self.init_mask = get_mask(ngf*8)
        self.scales = [4, 8, 16, 32, 64, 128, 256]

    @staticmethod
    def SAGB_progress(feat, conds, stage_mask, scale, SAGB_block):
        feat = F.interpolate(feat, scale_factor=2)
        stage_mask = F.interpolate(stage_mask, size=scale, mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        c_feat, stage_mask = SAGB_block(feat, conds, fusion_mask)
        return c_feat, stage_mask

    def forward(self, x, sent, attrs):
        # sent, attrs are the condition features

        out = self.fc(x)
        # 4 x 4
        out = out.view(x.size(0), 8 * self.ngf, 4, 4)
        stage_mask = self.init_mask(out)
        fusion_mask = torch.sigmoid(stage_mask)
        out, stage_mask = self.blocks[0](out, [sent, sent], fusion_mask)
        scales = [8, 16, 32]
        for ix, scale in enumerate(scales):
            out, stage_mask = \
                self.SAGB_progress(out, [sent, sent], stage_mask, scale, SAGB_block=self.blocks[ix+1])

        # 64 x 64
        x_32 = out
        x_64, stage_mask = \
            self.SARG_progress(x_32, [sent, attrs], stage_mask, 64, SAGB_block=self.blocks[4])
        x_128, stage_mask = \
            self.SARG_progress(x_64, [sent, attrs], stage_mask, 128, SAGB_block=self.blocks[5])
        x_256, _ = \
            self.SARG_progress(x_128, [sent, attrs], stage_mask, 256, SAGB_block=self.blocks[6])

        cum_x_64 = self.cum_64(x_32, x_64)
        cum_x_128 = self.cum_128(cum_x_64, x_128)
        cum_x_256 = self.cum_256(cum_x_128, x_256)

        img_64 = self.get_image_64(cum_x_64)
        img_128 = self.get_image_128(cum_x_128)
        img_256 = self.get_image_256(cum_x_256)

        return [img_64, img_128, img_256]

"""
The followings are designs of Discriminators
"""

class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)

class DiscSent(nn.Module):
    def __init__(self, ndf, nef):
        super(DiscSent, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf + nef, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False))

    def forward(self, feat, cond):
        cond = cond.view(-1, self.ef_dim, 1, 1)
        cond = cond.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((feat, cond), 1)
        out = self.joint_conv(h_c_code)
        return out

class DiscCond(nn.Module):
    def __init__(self, ndf, nef, class_nums=200):
        super(DiscCond, self).__init__()
        self.ndf = ndf
        self.nef = nef
        self.class_nums = class_nums

        self.joinConv = nn.Sequential(
            nn.Conv2d(ndf + nef, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.pair_node = nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=4)
        self.class_node = nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=4)
        self.class_linear = nn.Linear(ndf * 2, self.class_nums)

    def forward(self, img_code, c_code):

        # resize the condition
        scale = img_code.size(-1)
        c_code = c_code.view(-1, self.nef, 1, 1)
        c_code = c_code.repeat(1, 1, scale, scale)

        # merge the feature
        joint_code = torch.cat((img_code, c_code), 1)
        joint_code = self.joinConv(joint_code)

        pair_disc_out = self.pair_node(joint_code).view(-1)
        class_disc_out = self.class_node(joint_code).view(-1, self.ndf*2)
        class_disc_out = self.class_linear(class_disc_out)

        return pair_disc_out, class_disc_out

class Dis64(nn.Module):
    def __init__(self, ndf=cfg.GAN.DF_DIM):
        super(Dis64, self).__init__()
        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 64
        self.block0 = resD(ndf * 1, ndf * 2)  # 32
        self.block1 = resD(ndf * 2, ndf * 4)  # 16
        self.block2 = resD(ndf * 4, ndf * 8)  # 8
        self.block3 = resD(ndf * 8, ndf * 8)  # 4 x 4
        self.COND_DNET = DiscSent(ndf * 8, 256)

    def forward(self, x):
        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        return out

class Dis128(nn.Module):
    def __init__(self, ndf=cfg.GAN.DF_DIM):
        super(Dis128, self).__init__()
        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 128
        self.block0 = resD(ndf * 1, ndf * 2)  # 64
        self.block1 = resD(ndf * 2, ndf * 4)  # 32
        self.block2 = resD(ndf * 4, ndf * 8)  # 16
        self.block3 = resD(ndf * 8, ndf * 8)  # 4
        self.block4 = resD(ndf * 8, ndf * 16)  # 4
        self.COND_DNET = DiscSent(ndf * 16, 256)

    def forward(self, x):
        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return out

class Dis256(nn.Module):
    def __init__(self, ndf, disc_class, class_nums):
        super(Dis256, self).__init__()
        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 256
        self.block0 = resD(ndf * 1, ndf * 2)  # 128
        self.block1 = resD(ndf * 2, ndf * 4)  # 64
        self.block2 = resD(ndf * 4, ndf * 8)  # 32
        self.block3 = resD(ndf * 8, ndf * 16)  # 16
        self.block4 = resD(ndf * 16, ndf * 16)  # 8
        self.block5 = resD(ndf * 16, ndf * 16)  # 4
        self.disc_class = disc_class

        if disc_class:
            self.COND_DNET = DiscCond(ndf * 16, 256, class_nums=class_nums)
        else:
            self.COND_DNET = DiscSent(ndf * 16, 256)

    def forward(self, x):
        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return out

if __name__ == '__main__':
    # work
    from sync_batchnorm import DataParallelWithCallback
    gen = Gen(64, 100).to('cuda')
    gen = DataParallelWithCallback(gen)
    whole_syn_path = '../data/SSA_GAN_OUT/1_2/multi_attr/Model/netG_760.pth'
    state_dict = torch.load(whole_syn_path, map_location=lambda storage, loc: storage)
    gen.load_state_dict(state_dict)


