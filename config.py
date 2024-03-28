import argparse

args = argparse.ArgumentParser(description='config of VGG on fMRI')
args.add_argument('--feature_map', default='all', help='feature map of dataset--all, ALFF, fALFF, ReHo, VMHC')
args.add_argument('--model', default='VGGgap', help='name of model--VGG, VGGgap')
args.add_argument('--lr', type=float, default=5e-5)
args.add_argument('--epochs', type=int, default=200)
args.add_argument('--channels', type=int, default=16)
args.add_argument('--dropout', type=float, default=0.8)
args.add_argument('--weight_decay', type=float, default=5e-4)
args.add_argument('--scale', default='normal', help='scale of data--normal, zscore')
args.add_argument('--group', default='single', help='kind of dataset--single, group')
args.add_argument('--batch_size', type=int, default=4)
args.add_argument('--run_time', type=int, default=6)
args.add_argument('--dataset', default='ADNI', help='name of dataset--ADNI, COBRE')
args.add_argument('--kfold', type=int, default=10)

args = args.parse_args()
print(args)
