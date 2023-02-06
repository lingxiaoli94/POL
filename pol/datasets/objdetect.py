import torch
import fiftyone as fo
from PIL import Image
import numpy as np
import torchvision
import random

def get_square_padding(image, top_left=True):
    w, h = image.size
    max_wh = np.max([w, h])
    if top_left:
        h_padding = max_wh - w
        v_padding = max_wh - h
        l_pad = 0
        r_pad = h_padding
        t_pad = 0
        b_pad = v_padding
    else:
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class PadToSquare(object):
    def __init__(self, fill=0, top_left=False, padding_mode='constant'):
        self.fill = fill
        self.top_left = top_left
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return torchvision.transforms.functional.pad(
            img, get_square_padding(img, top_left=self.top_left), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)

class ResizeLonger(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        w, h = img.size
        if w > h:
            new_size = (self.length, self.length * h / w)
        else:
            new_size = (self.length * w / h, self.length)
        # height and width are reversed.
        new_size = (int(round(new_size[1])), int(round(new_size[0])))
        return torchvision.transforms.functional.resize(img, new_size)


class COCODataset(torch.utils.data.Dataset):
    def __init__(self,
                 split,
                 evaluation,
                 dataset_name,
                 persistent=True,
                 classes=None,
                 gt_field='ground_truth',
                 max_num_detection=None,
                 crop_size=400,
                 num_take=None,
                 take_seed=42,
                 keep_original=False):
        self.evaluation = evaluation
        self.gt_field = gt_field
        self.crop_size = crop_size
        self.keep_original = keep_original
        fo_dataset = fo.zoo.datasets.load_zoo_dataset(
            'coco-2017',
            dataset_name=dataset_name,
            split=split,
            label_types=['detections'],
            classes=classes,
        )
        fo_dataset = fo_dataset.match(
            fo.ViewField(f'{gt_field}.detections').length() > 0)
        if persistent:
            fo_dataset.persistent = True
        if max_num_detection is not None:
            fo_dataset = fo_dataset.match(
                fo.ViewField(f'{gt_field}.detections').length() <= max_num_detection)
        if num_take is not None:
            fo_dataset = fo_dataset.take(num_take, seed=take_seed)

        if classes is not None:
            self.class_set = set(classes)
        else:
            self.class_set = None

        from albumentations.pytorch import ToTensorV2
        import albumentations as A
        bbox_params = A.BboxParams(format='coco',
                                   min_visibility=0.3,
                                   label_fields=['labels'])
        if evaluation:
            pre_transforms = [
                A.RandomResizedCrop(crop_size, crop_size, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
            ]
            post_transforms = [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
            self.pre_transform = A.Compose(pre_transforms, bbox_params=bbox_params)
            self.post_transform = A.Compose(post_transforms, bbox_params=bbox_params)
            self.to_tensor_transform = ToTensorV2()
        else:
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomResizedCrop(crop_size, crop_size, scale=(0.2, 1.0)),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=bbox_params)

        self.img_paths = np.array(fo_dataset.values('filepath'))
        self.detections = fo_dataset.values(f'{gt_field}.detections')
        assert(len(self.detections) == len(self.img_paths))
        print(f'Loaded split {split} with {len(self.img_paths)} images.')

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        detections = self.detections[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        height, width, _ = img.shape

        boxes = []
        labels = []
        for d in detections:
            if self.class_set is not None and d.label not in self.class_set:
                continue
            x, y, w, h = d['bounding_box']
            if w < 1e-4 or h < 1e-4:
                continue # albumentation has issue with too narrow bounding boxes
            # (x, y) is top-left in coco format.
            # fiftyone use relative size of bounding boxes, so convert them to absolute size.
            boxes.append([x * width, y * height, w * width, h * height])
            labels.append(d.label)
        # assert(len(boxes) > 0)

        if self.evaluation:
            pre_transformed = self.pre_transform(image=img, bboxes=boxes, labels=labels)
            transformed = self.post_transform(image=pre_transformed['image'],
                                              bboxes=pre_transformed['bboxes'],
                                              labels=pre_transformed['labels'])
        else:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)

        trans_image = transformed['image']
        assert(trans_image.shape[1] == self.crop_size and
               trans_image.shape[2] == self.crop_size)
        trans_boxes = transformed['bboxes']
        # In our convention, bounding boxes are normalized with (x,y) centered.
        for i in range(len(trans_boxes)):
            x, y, w, h = trans_boxes[i]
            x = x / self.crop_size
            y = y / self.crop_size
            w = w / self.crop_size
            h = h / self.crop_size
            x += 0.5 * w
            y += 0.5 * h
            trans_boxes[i] = [x, y, w, h]

        result = {}
        result['img'] = transformed['image']
        result['boxes'] = torch.as_tensor(trans_boxes, dtype=torch.float32)
        result['labels'] = transformed['labels']
        if self.evaluation and self.keep_original:
            result['img_ori'] = self.to_tensor_transform(
                image=pre_transformed['image'])['image']
        return result

    def __len__(self):
        return len(self.img_paths)


def max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def collate_padding_fn(data_list):
    B = len(data_list)
    max_img_size = max_by_axis([list(data['img'].shape) for data in data_list])
    C, H, W = max_img_size
    max_num_box = torch.as_tensor(
        [data['boxes'].shape[0] for data in data_list]).max().item()
    max_num_box = max(1, max_num_box) # prevent all boxes empty
    dtype = torch.float32 # data_list[0]['img'].dtype
    device = data_list[0]['img'].device

    batched_img = torch.zeros([B, *max_img_size], dtype=dtype, device=device)
    if 'img_ori' in data_list[0]:
        batched_img_ori = torch.zeros([B, *max_img_size], dtype=dtype, device=device)
    batched_boxes = torch.zeros([B, max_num_box, 4], dtype=dtype, device=device)
    box_masks = torch.zeros([B, max_num_box], dtype=torch.bool, device=device)
    for i in range(len(data_list)):
        data = data_list[i]
        img = data['img']
        boxes = data['boxes']
        pad_img = batched_img[i]
        pad_boxes = batched_boxes[i]
        mask = box_masks[i]

        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        if boxes.shape[0] > 0:
            pad_boxes[:boxes.shape[0], :].copy_(boxes)
            mask[:boxes.shape[0]] = True
        if 'img_ori' in data_list[0]:
            pad_img_ori = batched_img_ori[i]
            img_ori = data['img_ori']
            pad_img_ori[:img_ori.shape[0],
                        :img_ori.shape[1],
                        :img_ori.shape[2]].copy_(img_ori)

    result = {'img': batched_img,
              'boxes': batched_boxes,
              'box_masks': box_masks}

    if 'img_ori' in data_list[0]:
        result['img_ori'] = batched_img_ori
    return result



