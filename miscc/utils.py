# -*- encoding: utf-8 -*-
import os
import errno
import shutil

import numpy as np
from torch.nn import init

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform

from miscc.config import cfg


def save_img_results(batch_imgs, prefix, image_dir, nrow=8):
    """
    parms: batch_imgs: bs x 3 x w x h
    """
    if isinstance(batch_imgs, list):
        for ix in range(len(batch_imgs)):
            fake_img = batch_imgs[ix]
            vutils.save_image(fake_img, "%s/%s_%d.png" % (image_dir, prefix, ix),
                              scale_each=True, normalize=True, nrow=nrow)
    else:
        vutils.save_image(batch_imgs, "%s/%s.png" % (image_dir, prefix),
                          scale_each=True, normalize=True, nrow=nrow)

def save_text_results(captions, cap_lens, ixtoword, txt_save_path,
                      attrs=None, attrs_num=None, attrs_len=None):
    """
    param: captions are the torch type
    """

    save_texts = list()
    for i in range(len(captions)):
        cap = captions[i].data.cpu().numpy()
        cap_len = cap_lens[i]
        words = [ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
                 for j in range(cap_len)]

        sent_str = " ".join(words)
        save_texts.append(sent_str)

        # attributes
        att_str = "# "
        # attrs_num[i] = 0, 1, 2, 3

        for attr_ix in range(attrs_num[i]):
            one_attr_len = attrs_len[i][attr_ix]
            one_attr = attrs[i][attr_ix].data.cpu().numpy()
            words = [ixtoword[one_attr[j]].encode('ascii', 'ignore').decode('ascii')
                     for j in range(one_attr_len)]
            att_str += " ".join(words) + ", "

        save_texts.append(att_str)

    f = open(txt_save_path, "w+")
    for one_line in save_texts:
        f.write(one_line + "\n")
    f.close()

def mkdir_p(path, rm_exist=False):
    try:
        if os.path.exists(path) and rm_exist:
            shutil.rmtree(path)
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise