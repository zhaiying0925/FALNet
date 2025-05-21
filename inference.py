import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
from tqdm import tqdm
from medpy.metric import binary
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim.swa_utils import AveragedModel
import random
from option import args
from dataset.Segmentation import DatasetSegmentationInfer
from model.Network import Network

import warnings
warnings.simplefilter("ignore", UserWarning)

def compute_dice(seg, gt):
    return binary.dc(seg, gt)


def compute_iou(s, g):
    intersection = np.logical_and(s, g).sum()
    union = np.logical_or(s, g).sum()
    return np.nanmean(intersection / (union + 1e-10))


def eval_dice(thres=0.5):
    dices, ious, res = [], [], []
    label_base = os.path.join(args.test_seg_dir, 'masks')
    pred_base = os.path.join(args.exp_dir, args.exp_name, args.save_masks)
    seg_type = '_polyp.png'
    for path in tqdm(os.listdir(label_base), desc="eval:"):
        label = os.path.join(label_base, path)
        pred = os.path.join(pred_base, path.split('.')[0] + seg_type)

        image2 = Image.open(pred)
        seg = np.array(image2)
        image1 = Image.open(label).convert('L')
        image1 = image1.resize((seg.shape[1], seg.shape[0]), Image.NEAREST)
        gt = np.array(image1)

        seg[seg >= 255 * thres] = 255
        seg[seg < 255 * thres] = 0

        Dice = compute_dice(seg, gt)
        iou = compute_iou(seg, gt)
        dices.append(Dice)
        ious.append(iou)

    return ["%.4f" % (sum(dices) / len(dices)), "%.4f" % (sum(ious) / len(ious))]


def evalResult(checkpoint_dir, eval_fig_dir, itr_range):
    [range0, range1] = itr_range
    os.makedirs(eval_fig_dir, exist_ok=True)
    Note = open(eval_fig_dir + '/preds_seg.txt', mode='w')
    iterations, dices, ious = [], [], []
    for model_path in sorted(os.listdir(checkpoint_dir)):
        iteration = int(model_path.split('.')[0])
        if range0 <= iteration <= range1:
            iterations.append(iteration)
            model_path = os.path.join(checkpoint_dir, model_path)
            [FPS_, FPS] = test(model_path)
            [Dice, IoU] = eval_dice(thres=0.5)
            Note.write("\nmodel:" + model_path.split("/")[-1] + ", mean Dice is " + str(Dice) + ', ' + "IoU is " + str(
                IoU) + ', FPS_:%.2f' % FPS_ + ' FPS:%.2f' % FPS)
            dices.append(float(Dice))
            ious.append(float(IoU))
            print("\n" + model_path.split("/")[-1] + " mean Dice is " + str(Dice) + ', ' + "IoU is " + str(
                IoU) + ' FPS_:%.2f' % FPS_ + ' FPS:%.2f' % FPS)
    Note.close()

    plt.plot(iterations, dices, 'bo--', alpha=0.5, linewidth=1, label='dice')
    plt.plot(iterations, ious, 'ro--', alpha=0.5, linewidth=1, label='iou')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('score')
    plt.savefig(eval_fig_dir + '/eval.jpg')


def matplotlib_multi_pic(index, res, out):
    fig, ax = plt.subplots(5, 4, figsize=(20, 27))
    for i in range(5):
        img = cv2.imread(res[i][0])
        img = cv2.resize(img, [args.image_size, args.image_size])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        polyp_seg = cv2.imread(res[i][1])
        highlight_seg_pred = cv2.imread(res[i][2])
        highlight_removal = cv2.imread(res[i][3])
        highlight_removal = cv2.cvtColor(highlight_removal, cv2.COLOR_BGR2RGB)
        ax[i][0].imshow(img)
        ax[i][0].set_title('img', fontsize=20)
        ax[i][1].imshow(polyp_seg)
        ax[i][1].set_title('polyp_seg', fontsize=20)
        ax[i][2].imshow(highlight_seg_pred)
        ax[i][2].set_title('highlight_seg_pred', fontsize=20)
        ax[i][3].imshow(highlight_removal)
        ax[i][3].set_title('highlight_removal', fontsize=20)
    plt.savefig(out + '/%d.png' % index)
    plt.close()


def randomDisplay(checkpoint_path, eval_fig_dir, num=5):
    test(checkpoint_path)
    displayPath = os.path.join(eval_fig_dir, 'randomDisplay')
    os.makedirs(displayPath, exist_ok=True)
    res = []
    image_base = os.path.join(args.test_seg_dir, 'images')
    pred_base = os.path.join(args.exp_dir, args.exp_name, args.save_masks)
    for path in os.listdir(image_base):
        image_path = os.path.join(image_base, path)
        polyp_seg_pred = os.path.join(pred_base, path.split('.')[0] + '_polyp.png')
        highlight_seg_pred = os.path.join(pred_base, path.split('.')[0] + '_highlight.png')
        # highlight_removal = os.path.join(pred_base, path.split('.')[0]+'_highlight_removal.png')
        res.append([image_path, polyp_seg_pred, highlight_seg_pred, image_path])
        # res.append([image_path, polyp_seg_pred, highlight_seg_pred, highlight_removal])
    np.random.seed(args.seed)
    idex = np.random.randint(len(res) - 1, size=num)
    for i in tqdm(range(num), desc="display:"):
        matplotlib_multi_pic(i, res[idex[i]:idex[i] + 5], displayPath)


def save_mask(pred, file):
    # pred = pred.permute(1, 2, 0)
    pred_np = pred.detach().cpu().numpy()
    pred_np = pred_np * 255
    pred_np = pred_np.astype(np.uint8)
    cv2.imwrite(file, pred_np)
    return


def test(checkpoint_file):
    # init
    save_dir = os.path.join(args.exp_dir, args.exp_name, args.save_masks)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # dataloader
    dataset_seg = DatasetSegmentationInfer(args, args.test_seg_dir)
    dataloader_seg = DataLoader(dataset_seg, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # network
    model = Network(args).cuda()

    model.load_state_dict(torch.load(checkpoint_file))
    model.eval()

    t0 = time.time()
    fps_save, fps_list, frame_num = [], [], 0  # mean fps
    for img, img_paths in tqdm(dataloader_seg):
        img = img.cuda()
        with torch.no_grad():
            t1 = time.time()
            outputs,_,_,_,_ = model(img)
            fps_list.append(img.size(0) / (time.time() - t1))  # mean fps
        polyp_mask = outputs

        for i, img_path in enumerate(img_paths):
            img_name = img_path.split('/')[-1]

            polyp_map = polyp_mask[i][0]
            polyp_file = os.path.join(save_dir, img_name.split('.')[0] + '_polyp.png')
            save_mask(polyp_map, polyp_file)

        frame_num += img.size(0)
    fps_save.append(frame_num / (time.time() - t0))
    return [sum(fps_list) / len(fps_list), sum(fps_save) / len(fps_save)]


if __name__ == '__main__':
    checkpoint_dir = os.path.join(args.exp_dir, args.exp_name, args.chekpoints)
    eval_fig_dir = os.path.join(args.exp_dir, args.exp_name, 'eval_fig')
    evalResult(checkpoint_dir, eval_fig_dir, itr_range=[0, args.iterations])  # 1000,args.iterations
