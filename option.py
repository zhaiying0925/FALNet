import argparse
parser = argparse.ArgumentParser(description='TRACER_AOTGAN')


# common
parser.add_argument('--exp_name', default='')
# last best 32 000 / 1250 000
parser.add_argument('--iterations', type=int, default=18125, help='the number of iterations for training')
parser.add_argument('--first_phase_iterations', type=int, default=0, help='the number of iterations for training first')
parser.add_argument('--seed', type=int, default=0, help='total seed')
parser.add_argument('--lr', type=float, default=5e-5) #5e-5
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
parser.add_argument('--exp_dir', default='exp_dir')
parser.add_argument('--chekpoints', default='chekpoints')
parser.add_argument('--save_masks', default='save_masks')
parser.add_argument('--save_iter', default=1000)

# backbone
parser.add_argument('--mscan_checkpoint', type=str, default='pretrained/mscan_t.pth')
parser.add_argument('--mscan', type=str, default='tiny')

# PS-KD
parser.add_argument('--PSKD', default=False,type=bool, help='PSKD')
parser.add_argument('--alpha_T',default=0.8,type=float, help='alpha_T')

# BotCL
parser.add_argument('--cpt_activation', default="att", help='the type to form cpt activation, default att using attention')
parser.add_argument('--weak_supervision_bias', type=float, default=0.1, help='weight fot the weak supervision branch')
parser.add_argument('--att_bias', type=float, default=0.1, help='used to prevent overflow, default as 0.1')
parser.add_argument('--quantity_bias', type=float, default=0.1, help='force each concept to be binary')
parser.add_argument('--distinctiveness_bias', type=float, default=0.01, help='refer to paper')
parser.add_argument('--consistence_bias', type=float, default=0.05, help='refer to paper')
parser.add_argument('--feature_size', default=10, help='size of the feature from backbone')
parser.add_argument('--is_vis', default=False, help='whether to visualize the concept')

# segmentation
parser.add_argument('--RFB_aggregated_channel', type=int, nargs='*', default=[32, 64, 128])
parser.add_argument('--denoise', type=float, default=0.93, help='Denoising background ratio')

# highlight removal
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406], help='')
parser.add_argument('--var', type=list, default=[0.229, 0.224, 0.225], help='')
parser.add_argument('--out_channel', type=int, default=3, help='total out channel')
parser.add_argument('--pixelShuffleRatio', type=int, default=2, help='the ratio of pixel shuffle')

# dataloader
parser.add_argument('--polyp_dir1', default='/root/')
parser.add_argument('--highlight_seg_dir', default='/root/')
parser.add_argument('--test_seg_dir', default='/root/')
parser.add_argument('--image_size', type=int, default=320, help='image size used during training')
parser.add_argument('--ver', type=int, default=2, help='type of transform')
parser.add_argument('--batch_size', type=int, default=8, help='batch size in each mini-batch') # 16
parser.add_argument('--num_workers', type=int, default=4, help='number of workers used in data loader')

parser.add_argument('--mix_prob', default=0.5, type=float,
                    help='mix probability')

args = parser.parse_args()
