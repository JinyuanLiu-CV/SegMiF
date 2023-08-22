import numpy as np
from torch.utils.data import Dataset
import os
import imageio
from . import imutils


def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list


class VOC12Dataset(Dataset):
    def __init__(
        self,
            root_dir=None,
            name_list_dir=None,
            split='train',
            stage='train',
            resize_range=[512, 360],
            rescale_range=[0.5, 2.0],
            crop_size=360,
            img_fliplr=True,
            ignore_index=255,
            aug=False,
            method = 'U2Fusion2',
            **kwargs
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir,method)
        self.label_dir = os.path.join(root_dir, 'Label')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = imutils.PhotoMetricDistortion()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name+'.png')
        image = np.asarray(imageio.imread(img_name))

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))
        #
        # elif self.stage == "test":
        #     label = image[:,:,0]

        return _img_name, image, label


class VOC12ClsDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 aug=False,
                 num_classes=21,
                 ignore_index=255,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            '''
            if self.resize_range:
                image, label = imutils.random_resize(
                    image, label, size_range=self.resize_range)
            '''
            if self.rescale_range:
                image, label = imutils.random_scaling(
                    image,
                    label,
                    scale_range=self.rescale_range,
                    size_range=self.resize_range)
            if self.img_fliplr:
                image, label = imutils.random_fliplr(image, label)
            if self.crop_size:
                image, label = imutils.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    mean_rgb=[123.675, 116.28, 103.53])

        image = imutils.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    @staticmethod
    def __to_onehot(label, num_classes):
        #label_onehot = F.one_hot(label, num_classes)
        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[label] = 1
        return label_onehot

    def __getitem__(self, idx):
        _img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)

        _label = np.unique(label).astype(np.int16)
        _label = _label[_label != self.ignore_index]
        #_label = _label[_label != 0]
        _label = self.__to_onehot(_label, self.num_classes)

        return _img_name, image, _label


class VOC12SegDataset(VOC12Dataset):
    def normalize_img__init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 method='U2Fusion2',
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage,method=method)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = imutils.PhotoMetricDistortion()

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        # print(np.shape(image), np.shape(label))

        if self.aug:
            '''
            if self.resize_range: 
                image, label = imutils.random_resize(
                    image, label, size_range=self.resize_range)
            '''
            if self.rescale_range:
                image, label = imutils.random_scaling(
                    image,
                    label,
                    scale_range=self.rescale_range,
                    size_range=self.resize_range)
            if self.img_fliplr:
                image, label = imutils.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = imutils.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    mean_rgb=[123.675, 116.28, 103.53], 
                    ignore_index=self.ignore_index)
        
        if self.stage != "train":
            image = imutils.img_resize_short(image, min_size=min(self.resize_range))

        image = imutils.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        _img_name, image, label = super().__getitem__(idx)
        if len( np.shape(image)) ==2:
            image = image[:, :, np.newaxis]
            image = np.concatenate([image,image,image],axis=2)
        image, label = self.__transforms(image=image, label=label)

        return _img_name, image, label
