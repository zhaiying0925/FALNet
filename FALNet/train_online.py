import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from option import args
from dataset.Segmentation import DatasetSegmentation, sample_data
from model.Network import Network
from loss.segmentation import adaptive_pixel_intensity_loss, CAMLoss
from loss.logit_margin_l1 import LogitMarginL1
from loss.BotCL import get_retrieval_loss, batch_cpt_discriminate, att_consistence, att_discriminate, att_binary, \
    att_area_loss
from tqdm import tqdm
from PIL import Image

import random
from skimage import segmentation
import copy
from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from torchsummary import summary
import cv2

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(0)
def worker_init_fn(worker_id):
    seed = 0 + worker_id
    np.random.seed(seed)
    random.seed(seed)
class Trainer():
    def __init__(self):
        super(Trainer, self).__init__()
        self.LogitMarginL1 = LogitMarginL1()
        self.cls_criterion = torch.nn.CrossEntropyLoss().cuda()
        self.CAMLoss = CAMLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.previous_batch = None

    def loss_seg(self, outputs, gt):
        loss1 = adaptive_pixel_intensity_loss(outputs, gt)
        loss5 = self.LogitMarginL1(outputs, gt)
        return loss1 + loss5

    def loss_seg_pred(self, outputs, gt):
        #gt = (gt > 0.5).float()
        outputs = outputs.unsqueeze(0)
        gt = gt.unsqueeze(0)
        loss1 = self.mse_loss(outputs, gt)
        loss5 = self.LogitMarginL1(outputs, gt)
        return loss1 + loss5

    def DeepSupervisionLoss(self, pred, gt):
        d0, d1, d2, d3, d4 = pred[0:]
        loss0 = adaptive_pixel_intensity_loss(d0, gt)
        loss0_L = self.LogitMarginL1(d0, gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss1 = adaptive_pixel_intensity_loss(d1, gt)
        loss1_L = self.LogitMarginL1(d1, gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss2 = adaptive_pixel_intensity_loss(d2, gt)
        loss2_L = self.LogitMarginL1(d2, gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss3 = adaptive_pixel_intensity_loss(d3, gt)
        loss3_L = self.LogitMarginL1(d3, gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss4 = adaptive_pixel_intensity_loss(d4, gt)
        loss4_L = self.LogitMarginL1(d4, gt)

        return loss0 + loss0_L + loss1 + loss1_L + loss2 + loss2_L + loss3 + loss3_L + loss4 + loss4_L

    def low_freq_mutate_tensor(self, amp_src, amp_trg, L=0.1):
        lam = 0.5
        a_src = torch.fft.fftshift(amp_src, dim=(-2, -1))
        a_trg = torch.fft.fftshift(amp_trg, dim=(-2, -1))

        _, h, w = a_src.shape
        b = int(torch.floor(torch.tensor(min(h, w), dtype=torch.float32) * L).item())
        c_h = h // 2
        c_w = w // 2

        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1

        a_src[:, h1:h2, w1:w2] = lam * a_trg[:, h1:h2, w1:w2] + (1 - lam) * a_src[:, h1:h2, w1:w2]
        a_src = torch.fft.ifftshift(a_src, dim=(-2, -1))
        return a_src

    def FDA_source_to_target_np(self, src_img, trg_img, L=0.1):
        fft_src = torch.fft.fft2(src_img, dim=(-2, -1))
        fft_trg = torch.fft.fft2(trg_img, dim=(-2, -1))

        amp_src = torch.abs(fft_src)
        pha_src = torch.angle(fft_src)
        amp_trg = torch.abs(fft_trg)
        pha_trg = torch.angle(fft_trg)
        amp_src_ = self.low_freq_mutate_tensor(amp_src, amp_trg, L=L)
        fft_src_ = amp_src_ * torch.exp(1j * pha_src)
        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1))
        src_in_trg = torch.real(src_in_trg)

        return src_in_trg

    def combine_three_images(self,original_img, augmented_img1, augmented_img2):
        height, width = original_img.shape[1], original_img.shape[2]
        combine_choice = random.choice(['horizontal', 'vertical'])

        if combine_choice == 'vertical':
            first_part_height = height // 3
            second_part_height = height // 3
            third_part_height = height - first_part_height - second_part_height

            mask = np.vstack([
                np.zeros((first_part_height, width)),
                0.5 * np.ones((second_part_height, width)),
                np.ones((third_part_height, width))
            ])

        else:
            first_part_width = width // 3
            second_part_width = width // 3
            third_part_width = width - first_part_width - second_part_width

            mask = np.hstack([
                np.zeros((height, first_part_width)),
                0.5 * np.ones((height, second_part_width)),
                np.ones((height, third_part_width))
            ])

        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        original_array = original_img.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        augmented_array1 = augmented_img1.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        augmented_array2 = augmented_img2.cpu().numpy().transpose(1, 2, 0).astype(np.float32)

        blended_array = (1 - mask) * original_array + mask * augmented_array1 * (
                    mask == 0.5) + mask * augmented_array2 * (mask == 1)

        return torch.tensor(blended_array).permute(2, 0, 1)

    def data_augmentation(self, img):
        aug_images = []
        if len(img) >= 3:
            self.previous_batch = img.clone()
        for i, img_a in enumerate(img):
            remaining_indices = list(range(len(img)))
            remaining_indices.remove(i)
            if len(remaining_indices) >= 2:
                img_b_idx, img_c_idx = random.sample(remaining_indices, 2)
                img_b, img_c = img[img_b_idx], img[img_c_idx]
            else:
                img_b, img_c = random.sample(list(self.previous_batch), 2)
            img_a_FDA_ab = self.FDA_source_to_target_np(img_a, img_b, L=0.001)
            img_a_FDA_ac = self.FDA_source_to_target_np(img_a, img_c, L=0.001)
            merged_img = self.combine_three_images(img_a, img_a_FDA_ab, img_a_FDA_ac)

            aug_images.append(merged_img)
        aug_images = torch.stack(aug_images, dim=0).type(torch.float32)
        return aug_images

    def train(self):

        # init
        checkpoint_dir = os.path.join(args.exp_dir, args.exp_name, args.chekpoints)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # dataloader
        phase = 'polyp'
        dataset_polyp_seg = DatasetSegmentation(args, phase, args.polyp_dir1)
        dataloader_polyp_seg = DataLoader(dataset_polyp_seg, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True)
        dataloader_polyp_seg = sample_data(dataloader_polyp_seg)

        # network
        model = Network(args)
        model = model.cuda()

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        pbar = tqdm(range(1, args.iterations + 1))
        # flag = 'polyp_seg'

        for itr in pbar:
            model.train()
            # PS-KD Alpha_t update
            if args.PSKD:
                alpha_t = args.alpha_T * (itr / args.iterations)
                alpha_t = max(0, alpha_t)
            else:
                alpha_t = -1
            # PS-KD load last model (single gpu)
            if args.PSKD and itr > args.save_iter and itr % args.save_iter == 0:
                last_model = Network(args).cuda()
                last_model_path = os.path.join(checkpoint_dir, f'{str(itr - args.save_iter).zfill(7)}.pth')
                last_model.load_state_dict(torch.load(last_model_path))
                last_model.eval()

            # train polyp segmentation
            img, gt = next(dataloader_polyp_seg)
            aug_gt = gt.clone()
            img, gt = img.cuda(), gt.cuda()
            aug_img = self.data_augmentation(img)
            aug_img, aug_gt = aug_img.cuda(), aug_gt.cuda()

            # PS_KD Self-KD or none
            if args.PSKD and itr > args.save_iter and itr % args.save_iter == 0:
                pred_pskd,_,_,_,_ = last_model(img)
                outputs_pskd = pred_pskd
                gt = ((1 - alpha_t) * gt) + (alpha_t * outputs_pskd)

            pred1 = model(img)
            pred2 = model(aug_img)
            optimizer.zero_grad()

            r = np.random.rand(1)
            if itr >= args.first_phase_iterations and r < args.mix_prob:
                loss1 = self.DeepSupervisionLoss(pred1, gt)
                loss2 = self.DeepSupervisionLoss(pred2, aug_gt)
                loss = 0.9*loss1 + 0.1*loss2
            else:
                loss1 = self.DeepSupervisionLoss(pred1, gt)
                loss = loss1

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clipping)
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            if itr % args.save_iter == 0 or itr == args.iterations:
                save_file = os.path.join(checkpoint_dir, f'{str(itr).zfill(7)}.pth')
                torch.save(model.state_dict(), save_file)
                print('checkpoint saved at: ', save_file)


if __name__ == '__main__':
    set_random_seed(args.seed)
    Trainer = Trainer()
    Trainer.train()
