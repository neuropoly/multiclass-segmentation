import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class DownConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        with warnings.catch_warnings(): # ignore the depreciation warning related to nn.Upsample
            warnings.simplefilter("ignore")
            x = self.up1(x)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class SmallUNet(nn.Module):
    def __init__(self, nb_input_channels, orientation, resolution, matrix_size,
                 class_names, drop_rate=0.4, bn_momentum=0.1, mean=0., std=1.):
        super(SmallUNet, self).__init__()

        self.mean = mean
        self.std = std
        self.orientation = orientation
        self.resolution = resolution
        self.matrix_size = matrix_size
        self.class_names = class_names
        nb_classes = 1
        if len(class_names)>1:
            nb_classes=len(class_names)+1

        #Downsampling path
        self.conv1 = DownConv(nb_input_channels, 32, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(32, 64, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottom
        self.conv4 = DownConv(128, 128, drop_rate, bn_momentum)

        # Upsampling path
        self.up1 = UpConv(256, 128, drop_rate, bn_momentum)
        self.up2 = UpConv(192, 64, drop_rate, bn_momentum)
        self.up3 = UpConv(96, 32, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(32, nb_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = (x-self.mean)/self.std

        x1 = self.conv1(x)
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        x6 = self.mp3(x5)

        # Bottom
        x7 = self.conv4(x6)

        # Up-sampling
        x8 = self.up1(x7, x5)
        x9 = self.up2(x8, x3)
        x10 = self.up3(x9, x1)

        x11 = self.conv9(x10)
        if len(self.class_names)>1:
            preds = F.softmax(x11, 1)
        else:
            preds = F.sigmoid(x11)

        return preds


class NoPoolASPP(nn.Module):
    """
    .. image:: _static/img/nopool_aspp_arch.png
        :align: center
        :scale: 25%
    An ASPP-based model without initial pooling layers.
    :param drop_rate: dropout rate.
    :param bn_momentum: batch normalization momentum.
    .. seealso::
        Perone, C. S., et al (2017). Spinal cord gray matter
        segmentation using deep dilated convolutions.
        Nature Scientific Reports link:
        https://www.nature.com/articles/s41598-018-24304-3
    """
    def __init__(self, nb_input_channels, mean, std, orientation, resolution,
                 matrix_size, class_names, drop_rate=0.4, bn_momentum=0.1, base_num_filters=64):
        super(NoPoolASPP, self).__init__()

        self.mean = mean
        self.std = std
        self.orientation = orientation
        self.resolution = resolution
        self.matrix_size = matrix_size
        self.class_names = class_names
        nb_classes = 1
        if len(class_names)>1:
            nb_classes=len(class_names)+1

        self.conv1a = nn.Conv2d(nb_input_channels, base_num_filters, kernel_size=3, padding=1)
        self.conv1a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv1a_drop = nn.Dropout2d(drop_rate)
        self.conv1b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=1)
        self.conv1b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv1b_drop = nn.Dropout2d(drop_rate)

        self.conv2a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=2, dilation=2)
        self.conv2a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv2a_drop = nn.Dropout2d(drop_rate)
        self.conv2b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=2, dilation=2)
        self.conv2b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv2b_drop = nn.Dropout2d(drop_rate)

        # Branch 1x1 convolution
        self.branch1a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=1)
        self.branch1a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch1a_drop = nn.Dropout2d(drop_rate)
        self.branch1b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=1)
        self.branch1b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch1b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 6
        self.branch2a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=6, dilation=6)
        self.branch2a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch2a_drop = nn.Dropout2d(drop_rate)
        self.branch2b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=6, dilation=6)
        self.branch2b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch2b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 12
        self.branch3a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=12, dilation=12)
        self.branch3a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch3a_drop = nn.Dropout2d(drop_rate)
        self.branch3b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=12, dilation=12)
        self.branch3b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch3b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 18
        self.branch4a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=18, dilation=18)
        self.branch4a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch4a_drop = nn.Dropout2d(drop_rate)
        self.branch4b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=18, dilation=18)
        self.branch4b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch4b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 24
        self.branch5a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=24, dilation=24)
        self.branch5a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch5a_drop = nn.Dropout2d(drop_rate)
        self.branch5b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=24, dilation=24)
        self.branch5b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch5b_drop = nn.Dropout2d(drop_rate)

        self.concat_drop = nn.Dropout2d(drop_rate)
        self.concat_bn = nn.BatchNorm2d(6*base_num_filters, momentum=bn_momentum)

        self.amort = nn.Conv2d(6*base_num_filters, base_num_filters*2, kernel_size=1)
        self.amort_bn = nn.BatchNorm2d(base_num_filters*2, momentum=bn_momentum)
        self.amort_drop = nn.Dropout2d(drop_rate)

        self.prediction = nn.Conv2d(base_num_filters*2, nb_classes, kernel_size=1)

    def forward(self, x):
        """Model forward pass.
        :param x: input data.
        """
        x = (x-self.mean)/self.std

        x = F.relu(self.conv1a(x))
        x = self.conv1a_bn(x)
        x = self.conv1a_drop(x)

        x = F.relu(self.conv1b(x))
        x = self.conv1b_bn(x)
        x = self.conv1b_drop(x)

        x = F.relu(self.conv2a(x))
        x = self.conv2a_bn(x)
        x = self.conv2a_drop(x)
        x = F.relu(self.conv2b(x))
        x = self.conv2b_bn(x)
        x = self.conv2b_drop(x)

        # Branch 1x1 convolution
        branch1 = F.relu(self.branch1a(x))
        branch1 = self.branch1a_bn(branch1)
        branch1 = self.branch1a_drop(branch1)
        branch1 = F.relu(self.branch1b(branch1))
        branch1 = self.branch1b_bn(branch1)
        branch1 = self.branch1b_drop(branch1)

        # Branch for 3x3 rate 6
        branch2 = F.relu(self.branch2a(x))
        branch2 = self.branch2a_bn(branch2)
        branch2 = self.branch2a_drop(branch2)
        branch2 = F.relu(self.branch2b(branch2))
        branch2 = self.branch2b_bn(branch2)
        branch2 = self.branch2b_drop(branch2)

        # Branch for 3x3 rate 6
        branch3 = F.relu(self.branch3a(x))
        branch3 = self.branch3a_bn(branch3)
        branch3 = self.branch3a_drop(branch3)
        branch3 = F.relu(self.branch3b(branch3))
        branch3 = self.branch3b_bn(branch3)
        branch3 = self.branch3b_drop(branch3)

        # Branch for 3x3 rate 18
        branch4 = F.relu(self.branch4a(x))
        branch4 = self.branch4a_bn(branch4)
        branch4 = self.branch4a_drop(branch4)
        branch4 = F.relu(self.branch4b(branch4))
        branch4 = self.branch4b_bn(branch4)
        branch4 = self.branch4b_drop(branch4)

        # Branch for 3x3 rate 24
        branch5 = F.relu(self.branch5a(x))
        branch5 = self.branch5a_bn(branch5)
        branch5 = self.branch5a_drop(branch5)
        branch5 = F.relu(self.branch5b(branch5))
        branch5 = self.branch5b_bn(branch5)
        branch5 = self.branch5b_drop(branch5)

        # Global Average Pooling
        global_pool = F.avg_pool2d(x, kernel_size=x.size()[2:])
        global_pool = global_pool.expand(x.size())

        concatenation = torch.cat([branch1, branch2, branch3, branch4, branch5, global_pool], dim=1)

        concatenation = self.concat_bn(concatenation)
        concatenation = self.concat_drop(concatenation)

        amort = F.relu(self.amort(concatenation))
        amort = self.amort_bn(amort)
        amort = self.amort_drop(amort)

        predictions = self.prediction(amort)
        predictions = F.sigmoid(predictions)

        return predictions


class SegNet(nn.Module):
    """Segnet network."""

    def __init__(self, nb_input_channels, class_names, mean, std, orientation,
                 resolution, matrix_size, bn_momentum=0.1, drop_rate=0.4):
        """Init fields."""
        super(SegNet, self).__init__()

        self.input_nbr = nb_input_channels
        self.mean = mean
        self.std = std
        self.orientation = orientation
        self.resolution = resolution
        self.matrix_size = matrix_size
        self.class_names = class_names
        label_nbr = 1
        if len(class_names)>1:
            label_nbr=len(class_names)+1


        self.conv11 = nn.Conv2d(nb_input_channels, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.drop11 = nn.Dropout2d(drop_rate)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.drop12 = nn.Dropout2d(drop_rate)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.drop21 = nn.Dropout2d(drop_rate)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.drop22 = nn.Dropout2d(drop_rate)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop31 = nn.Dropout2d(drop_rate)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop32 = nn.Dropout2d(drop_rate)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop33 = nn.Dropout2d(drop_rate)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop41 = nn.Dropout2d(drop_rate)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop42 = nn.Dropout2d(drop_rate)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop43 = nn.Dropout2d(drop_rate)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop51 = nn.Dropout2d(drop_rate)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop52 = nn.Dropout2d(drop_rate)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop53 = nn.Dropout2d(drop_rate)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop53d = nn.Dropout2d(drop_rate)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop52d = nn.Dropout2d(drop_rate)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop51d = nn.Dropout2d(drop_rate)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop43d = nn.Dropout2d(drop_rate)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop42d = nn.Dropout2d(drop_rate)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop41d = nn.Dropout2d(drop_rate)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop33d = nn.Dropout2d(drop_rate)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop32d = nn.Dropout2d(drop_rate)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.drop31d = nn.Dropout2d(drop_rate)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.drop22d = nn.Dropout2d(drop_rate)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.drop21d = nn.Dropout2d(drop_rate)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.drop12d = nn.Dropout2d(drop_rate)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward method."""
        # normalization
        x = (x-self.mean)/self.std

        # Stage 1
        x11 = F.relu(self.drop11(self.bn11(self.conv11(x))))
        x12 = F.relu(self.drop12(self.bn12(self.conv12(x11))))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)
        size1 = x12.size()

        # Stage 2
        x21 = F.relu(self.drop21(self.bn21(self.conv21(x1p))))
        x22 = F.relu(self.drop22(self.bn22(self.conv22(x21))))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)
        size2 = x22.size()
        # Stage 3
        x31 = F.relu(self.drop31(self.bn31(self.conv31(x2p))))
        x32 = F.relu(self.drop32(self.bn32(self.conv32(x31))))
        x33 = F.relu(self.drop33(self.bn33(self.conv33(x32))))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)
        size3 = x33.size()

        # Stage 4
        x41 = F.relu(self.drop41(self.bn41(self.conv41(x3p))))
        x42 = F.relu(self.drop42(self.bn42(self.conv42(x41))))
        x43 = F.relu(self.drop43(self.bn43(self.conv43(x42))))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)
        size4 = x43.size()

        # Stage 5
        x51 = F.relu(self.drop51(self.bn51(self.conv51(x4p))))
        x52 = F.relu(self.drop52(self.bn52(self.conv52(x51))))
        x53 = F.relu(self.drop53(self.bn53(self.conv53(x52))))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)
        size5 = x53.size()

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=size5)
        x53d = F.relu(self.drop53d(self.bn53d(self.conv53d(x5d))))
        x52d = F.relu(self.drop52d(self.bn52d(self.conv52d(x53d))))
        x51d = F.relu(self.drop51d(self.bn51d(self.conv51d(x52d))))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=size4)
        x43d = F.relu(self.drop43d(self.bn43d(self.conv43d(x4d))))
        x42d = F.relu(self.drop42d(self.bn42d(self.conv42d(x43d))))
        x41d = F.relu(self.drop41d(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=size3)
        x33d = F.relu(self.drop33d(self.bn33d(self.conv33d(x3d))))
        x32d = F.relu(self.drop32d(self.bn32d(self.conv32d(x33d))))
        x31d = F.relu(self.drop31d(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=size2)
        x22d = F.relu(self.drop22d(self.bn22d(self.conv22d(x2d))))
        x21d = F.relu(self.drop21d(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=size1)
        x12d = F.relu(self.drop12d(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d


class UNet(nn.Module):
    def __init__(self, nb_input_channels, orientation, resolution, matrix_size,
                 class_names, drop_rate=0.4, bn_momentum=0.1, mean=0., std=1.):
        super(UNet, self).__init__()

        self.mean = mean
        self.std = std
        self.orientation = orientation
        self.resolution = resolution
        self.matrix_size = matrix_size
        self.class_names = class_names
        nb_classes = 1
        if len(class_names)>1:
            nb_classes=len(class_names)+1

        #Downsampling path
        self.conv1 = DownConv(nb_input_channels, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        self.conv4 = DownConv(256, 512, drop_rate, bn_momentum)
        self.mp4 = nn.MaxPool2d(2)

        # Bottom
        self.conv5 = DownConv(512, 512, drop_rate, bn_momentum)

        # Upsampling path
        self.up1 = UpConv(1024, 512, drop_rate, bn_momentum)
        self.up2 = UpConv(768, 256, drop_rate, bn_momentum)
        self.up3 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up4 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv11 = nn.Conv2d(64, nb_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = (x-self.mean)/self.std

        x1 = self.conv1(x)
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        x6 = self.mp3(x5)

        x7 = self.conv4(x6)
        x8 = self.mp4(x7)

        # Bottom
        x9 = self.conv5(x8)

        # Up-sampling
        x10 = self.up1(x9, x7)
        x11 = self.up2(x10, x5)
        x12 = self.up3(x11, x3)
        x13 = self.up4(x12, x1)

        x14 = self.conv11(x13)

        if len(self.class_names)>1:
            preds = F.softmax(x14, 1)
        else:
            preds = F.sigmoid(x14)

        return preds
