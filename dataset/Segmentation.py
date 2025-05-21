import cv2
import torch
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
import os


def get_train_augmentation(img_size, ver):
    if ver == 1:
        transforms = albu.Compose([
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    if ver == 2:
        transforms = albu.Compose([
            albu.OneOf([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomRotate90()
            ], p=0.5),
            albu.OneOf([
                albu.RandomBrightnessContrast(),
                albu.RandomGamma(),
                #albu.RandomBrightness(),
            ], p=0.5),
            albu.OneOf([
                albu.MotionBlur(blur_limit=5),
                albu.MedianBlur(blur_limit=5),
                albu.GaussianBlur(blur_limit=5),
                albu.GaussNoise(var_limit=(5.0, 20.0)),
            ], p=0.5),
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return transforms

def get_test_augmentation(img_size):
    return albu.Compose(
        [
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def gt_to_tensor(gt):
    gt = cv2.imread(gt)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) / 255.0
    gt = np.where(gt > 0.5, 1.0, 0.0)
    gt = torch.tensor(gt, device='cuda', dtype=torch.float32)
    gt = gt.unsqueeze(0).unsqueeze(1)

    return gt

def sample_data(loader):
    while True:
        yield from loader

class DatasetSegmentation(Dataset):
    def __init__(self, args, phase, fol_dir):
        self.phase = phase
        self.gts, self.images = [], []

        img_path = os.path.join(fol_dir, 'images')
        gt_path = os.path.join(fol_dir, 'masks')
        img_list = os.listdir(img_path)
        for img_name in img_list:
            self.images.append(os.path.join(img_path, img_name))
            self.gts.append(os.path.join(gt_path, img_name))

        self.transform = get_train_augmentation(args.image_size, ver=args.ver)


    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.gts[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #classes = [1.,0.] if cv2.countNonZero(mask) == 0 else [0.,1.]
        if self.transform is not None:
            augmented = self.transform(image=image, masks=[mask])
            image = augmented['image']
            mask = augmented['masks'][0]  # (1, H, W)
            mask = mask / 255.0
            mask = torch.unsqueeze(mask, dim=0)
            mask = mask.type_as(image)

        return image, mask

    def __len__(self):
        return len(self.images)


class DatasetSegmentationInfer(Dataset):
    def __init__(self, args, fol_dir):
        img_path = os.path.join(fol_dir, 'images')
        img_list = os.listdir(img_path)

        self.images = []
        self.images.extend(os.path.join(img_path, img_name) for img_name in img_list)
        self.transform = get_test_augmentation(args.image_size)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.images[idx]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    from option import args
    from torch.utils.data import DataLoader

    phase = 'train'
    fol_dir = '/data/'
    dataset = DatasetSegmentation(args, phase, fol_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    for img, mask in data_loader:
        print(img.size(), mask.size())
