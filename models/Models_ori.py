import torch.nn as nn
import torch
import pywt

import torch.nn.functional as F
from functools import partial
from torch.nn import init
from models.dpcd_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, DPFA, Decoder_Block,
                               Changer_channel_exchange, log_feature, ABconv)

from models.hfa import SAFCB
from models.wtb import WTAB, WTAB1


class DPCD(nn.Module):
    """ Change detection model

    Input :obj:`t1_img` and :obj:`t2_img`, extract encoder feature by :obj:`en_block1-4`,
    then exchange channel feature of :obj:`t1_feature` and :obj:`t2_feature`, and extract
    encoder feature by :obj:`en_block5`.

    Upsample to get decoder feature by :obj:`de_block1-3`, get :obj:`seg_feature1` and :obj:`seg_feature2`
    by :obj:`seg_out1` and :obj:`seg_out2`.

    Fuse t1 and t2 corresponding feature to get change feature by :obj:`dpfa` and :obj:`change_blcok`.

    Notice that output of module and model could be log in this model.

    Attribute:
        en_block(class): encoder feature extractor.
        channel_exchange(class): exchange t1 and t2 feature.
        de_block(class): decoder feature upsampler and extractor.
        dpfa(class): fuse t1 and t2 feature to get change feature
            using both spatial and channel attention.
        change_block(class): change feature upsampler and extracor.
        seg_out(class): get decoder feature seg out result.
        upsample_x2(class): upsample change feature by 2.
        conv_out_change(class): conv out change feature out result.
    """

    def __init__(self, device, channel_list, embeding_dims):
        super().__init__()

        self.device = device
        self.channel_list = channel_list
        self.embeding_dims = embeding_dims
        # encoder
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=self.channel_list[0], kernel=3, stride=1),
                                       CGSU(in_channel=self.channel_list[0]),
                                       CGSU(in_channel=self.channel_list[0]),
                                       )
        self.en_block2 = Encoder_Block(in_channel=self.channel_list[0], out_channel=self.channel_list[1])
        self.en_block3 = Encoder_Block(in_channel=self.channel_list[1], out_channel=self.channel_list[2])
        self.en_block4 = Encoder_Block(in_channel=self.channel_list[2], out_channel=self.channel_list[3])
        self.en_block5 = Encoder_Block(in_channel=self.channel_list[3], out_channel=self.channel_list[4])

        self.channel_exchange4 = Changer_channel_exchange()

        # decoder
        self.de_block1 = Decoder_Block(in_channel=self.channel_list[4], out_channel=self.channel_list[3])
        self.de_block2 = Decoder_Block(in_channel=self.channel_list[3], out_channel=self.channel_list[2])
        self.de_block3 = Decoder_Block(in_channel=self.channel_list[2], out_channel=self.channel_list[1])

        # dpfa
        self.dpfa1 = DPFA(in_channel=self.channel_list[4])
        # self.dpfa2 = DPFA(in_channel=self.channel_list[3])
        # self.dpfa3 = DPFA(in_channel=self.channel_list[2])
        # self.dpfa4 = DPFA(in_channel=self.channel_list[1])

        self.dpfa2 = WTAB1(in_chans=channel_list[3], out_chans=channel_list[3], kernel_size=3, wavelet='haar',
                         device = self.device)
        self.dpfa3 = WTAB1(in_chans=channel_list[2], out_chans=channel_list[2], kernel_size=3, wavelet='haar',
                           device=self.device)
        self.dpfa4 = WTAB1(in_chans=channel_list[1], out_chans=channel_list[1], kernel_size=3, wavelet='haar',
                           device=self.device)

        # change path
        # the change block is the same as decoder block
        # the change block is used to fuse former and latter change features
        self.change_block4 = Decoder_Block(in_channel=self.channel_list[4], out_channel=self.channel_list[3])
        self.change_block3 = Decoder_Block(in_channel=self.channel_list[3], out_channel=self.channel_list[2])
        self.change_block2 = Decoder_Block(in_channel=self.channel_list[2], out_channel=self.channel_list[1])

        self.seg_out1 = nn.Conv2d(self.channel_list[1], 1, kernel_size=3, stride=1, padding=1)
        self.seg_out2 = nn.Conv2d(self.channel_list[1], 1, kernel_size=3, stride=1, padding=1)

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(self.channel_list[1], self.channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_change = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)
        #
        self.SAF = SAFCB(in_chans=3, out_chans=3, embed_dims=self.embeding_dims,device=self.device)

        self.WTA1 = WTAB(in_chans=channel_list[0], out_chans=channel_list[0], kernel_size=3, wavelet='haar',
                         device = self.device)
        self.WTA2 = WTAB(in_chans=channel_list[1], out_chans=channel_list[1], kernel_size=3, wavelet='haar',
                         device=self.device)
        self.WTA3 = WTAB(in_chans=channel_list[2], out_chans=channel_list[2], kernel_size=3, wavelet='haar',
                         device=self.device)


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')

                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def image_random_exchange(self, image1, image2):
        b, h, w, c = image1.shape
        l1 = []
        l2 = []
        for i in range(b):
            temp1 = image1[i].cuda()
            temp2 = image2[i].cuda()

            mask = torch.zeros(h, w).cuda()
            mask.bernoulli_(0.5)
            mask = torch.reshape(mask, (h, w, 1))
            mask = mask.bool()

            out1 = torch.add(torch.mul(temp1, mask), torch.mul(temp2, ~mask))
            out2 = torch.add(torch.mul(temp2, mask), torch.mul(temp1, ~mask))

            l1.append(out1)
            l2.append(out2)

        out1 = l1[-1].unsqueeze(dim=0)
        out2 = l2[-1].unsqueeze(dim=0)
        for i in range(len(l1) - 1):
            nn = l1[i].unsqueeze(dim=0)
            out1 = torch.cat((out1, nn), dim=0)
            nn = l2[i].unsqueeze(dim=0)
            out2 = torch.cat((out2, nn), dim=0)

        return out1, out2
    def forward(self, t1, t2, log=False, img_name=None, scale_ratio = None):
        """ Model forward and log feature if :obj:`log: is True.

        If :obj:`log` is True, some module output and model output will be saved.

        To be more specific, module output will be saved in folder named
        `:obj:`module_input_feature_name`_:obj:`module_name`-
        :obj:`module_input_feature_name`_:obj:`module_name`- ...
        _:obj:`log_feature_name``. For example, module output saved folder could be named
        `t1_1_en_block2-x_cbam-spatial_weight`.

        Module output in saved folder will have the same name as corresponding input image.

        Model output saved folder will be simply named `model_:obj:`log_feature_name``. For example,
        it could be `model_seg_out_1`.

        Model output in saved folder will have the same name as corresponding input image.

        :obj:`seg_out1` and :obj:`seg_out2` could be used in loss function to train model better,
        and :obj:`change_out` is the prediction of model.

        Parameter:
            t1(tensor): input t1 image.
            t2(tensor): input t2 image.
            log(bool): if True, log output of module and model.
            img_name(tensor): name of input image.

        Return:
            change_out(tensor): change prediction of model.
            seg_out1(tensor): auxiliary change prediction through t1 decoder branch.
            seg_out2(tensor): auxiliary change prediction through t2 decoder branch.
        """




        t1, t2, hx2_ffted = self.SAF(t1, t2)

        t1_1 = self.en_block1(t1)
        t2_1 = self.en_block1(t2)
        t1_1, t2_1 = self.WTA1(t1_1, t2_1)

        t1_2 = self.en_block2(t1_1)
        t2_2 = self.en_block2(t2_1)
        t1_2, t2_2 = self.WTA2(t1_2, t2_2)


        t1_3 = self.en_block3(t1_2)
        t2_3 = self.en_block3(t2_2)
        t1_3, t2_3 = self.WTA3(t1_3, t2_3)

        t1_4 = self.en_block4(t1_3)
        t2_4 = self.en_block4(t2_3)
        t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)


        t1_5 = self.en_block5(t1_4)
        t2_5 = self.en_block5(t2_4)


        de1_5 = t1_5
        de2_5 = t2_5

        de1_4 = self.de_block1(de1_5, t1_4)
        de2_4 = self.de_block1(de2_5, t2_4)

        de1_3 = self.de_block2(de1_4, t1_3)
        de2_3 = self.de_block2(de2_4, t2_3)

        de1_2 = self.de_block3(de1_3, t1_2)
        de2_2 = self.de_block3(de2_3, t2_2)

        seg_out1 = self.seg_out1(de1_2)
        seg_out2 = self.seg_out2(de2_2)

        # if log:
        #     change_5 = self.dpfa1(de1_5, de2_5, log=log, module_name='de1_5_de2_5_dpfa1',
        #                           img_name=img_name)
        #
        #     change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4, log=log, module_name='de1_4_de2_4_dpfa2',
        #                                                        img_name=img_name))
        #
        #     change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3, log=log, module_name='de1_3_de2_3_dpfa3',
        #                                                        img_name=img_name))
        #
        #     change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2, log=log, module_name='de1_2_de2_2_dpfa4',
        #                                                        img_name=img_name))

        change_5 = self.dpfa1(de1_5, de2_5)

        change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4))

        change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3))

        change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2))

        change = self.upsample_x2(change_2)
        change_out = self.conv_out_change(change)

        if log:
            log_feature(log_list=[change_out, seg_out1, seg_out2], module_name='model',
                        feature_name_list=['change_out', 'seg_out1', 'seg_out2'],
                        img_name=img_name, module_output=False,
                        scale_ratio = scale_ratio)

        return change_out, seg_out1, seg_out2
