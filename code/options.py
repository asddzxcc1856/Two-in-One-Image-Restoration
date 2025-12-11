import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int, default=2,
                    help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['derain'],
                    help='which type of degradations is training for.')

parser.add_argument('--patch_size', type=int, default=192,
                    help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=32,
                    help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--desnow_dir', type=str, default='data/Train/Desnow/',
                    help='where training images of desnowing saves.')
parser.add_argument('--output_path', type=str, default="output/",
                    help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/",
                    help='checkpoint save path')
parser.add_argument("--wblogger", type=str, default="promptir",
                    help="Determine to log to wandb project name")
parser.add_argument("--ckpt_dir", type=str, default="train_ckpt",
                    help="Where the checkpoint is to be saved")
parser.add_argument("--num_gpus", type=int, default=1,
                    help="Number of GPUs to use for training")

parser.add_argument("--use_cutmix", type=int, default=True,
                    help="Whether to apply CutMix data augmentation (0 or 1)")
parser.add_argument("--use_mixup", type=bool, default=False,
                    help="Whether to apply MixUp data augmentation")
parser.add_argument("--mixup_alpha", type=float, default=0.2,
                    help="Alpha value for MixUp (controls interpolation strength)")

options = parser.parse_args()
