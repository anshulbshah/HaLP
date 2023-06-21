import argparse
import math
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.distributed as dist
import moco.builder_single_modality_baseline
import moco.builder_single_modality_halp

from dataset import get_pretraining_set_intra, get_finetune_training_set, get_finetune_validation_set
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from termcolor import cprint
from pathlib import Path
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[100, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Distributed
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--checkpoint-path', default='./checkpoints', type=str)
parser.add_argument('--skeleton-representation', type=str,
                    help='input skeleton-representation  for self supervised training (image-based or graph-based or seq-based)')
parser.add_argument('--pre-dataset', default='ntu60', type=str,
                    help='which dataset to use for self supervised training (ntu60 or ntu120)')
parser.add_argument('--protocol', default='cross_subject', type=str,
                    help='traiining protocol cross_view/cross_subject/cross_setup')

# contrast specific configs:
parser.add_argument('--contrast-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--contrast-k', default=32768, type=int,
                    help='queue size; number of negative keys (default: 16384)')
parser.add_argument('--contrast-m', default=0.999, type=float,
                    help='contrast momentum of updating key encoder (default: 0.999)')
parser.add_argument('--contrast-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--teacher-t', default=0.05, type=float,
                    help='softmax temperature (default: 0.05)')
parser.add_argument('--student-t', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')
parser.add_argument('--cmd-weight', default=1.0, type=float,
                    help='weight of sim loss (default: 1.0)')
parser.add_argument('--topk', default=1024, type=int,
                    help='number of contrastive context')
parser.add_argument('--knn_every', default=5, type=int,
                    help='number of contrastive context')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--save_every', default=10, type=int,
                    help='number of contrastive context')

# HaLP specific options
parser.add_argument('--modality_to_use', default='joint', type=str, help='mod to use')
parser.add_argument('--method', default='baseline', type=str, help='mod to use')
parser.add_argument('--num_prototypes', default=20, type=int, help='Num prototypes to use')
parser.add_argument('--update_prototypes_every', default=5, type=int, help='Update prototypes every')
parser.add_argument('--num_positives', default=100, type=int, help='#Pos')
parser.add_argument("--num_closest_to_ignore_positives", type=int, default=20)
parser.add_argument('--lambda_pos', default="0.8", type=str, help='lambda pos max')
parser.add_argument('--mu', default=1.0, type=float,help='weight of halp loss (default: 1.0)')
parser.add_argument('--queue_els_for_prototypes', default=256, type=int, help='Number of queue elements to use for clustering')
parser.add_argument('--skip_knn', action='store_true',help='Skip kNN evaluation')
parser.add_argument("--skip_closest_positives", type=int, default=1)


def init_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    cudnn.deterministic = True
    cudnn.benchmark = True

def main():
    args = parser.parse_args()

    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)

    wandb.init(project="halp",config=args,tags=['pretrain'],settings=wandb.Settings(start_method="fork"))
    if wandb.run.sweep_id is None:
        sweep_name = 'no_sweep'
    else:
        sweep_name = wandb.run.sweep_id
    args.checkpoint_path = f'checkpoints/{sweep_name}/{wandb.run.name}'

    # Simply call main_worker function
    main_worker(args)

def test_extract_hidden(model, data_train, data_eval):
    model.eval()
    print("Extracting training features")
    label_train_list = []
    hidden_array_train_list = []
    for ith, (ith_data, label) in enumerate(tqdm(data_train)):
        input_tensor = ith_data.cuda()

        with torch.no_grad():
            en_hi = model(input_tensor, view='joint', knn_eval=True)
        en_hi = en_hi.squeeze()

        label_train_list.append(label)
        hidden_array_train_list.append(en_hi[:, :].detach().cpu().numpy())
    label_train = np.hstack(label_train_list)
    hidden_array_train = np.vstack(hidden_array_train_list)

    print("Extracting validation features")
    label_eval_list = []
    hidden_array_eval_list = []
    for ith, (ith_data,  label) in enumerate(tqdm(data_eval)):

        input_tensor = ith_data.cuda()

        with torch.no_grad():
            en_hi = model(input_tensor, view='joint', knn_eval=True)
        en_hi = en_hi.squeeze()

        label_eval_list.append(label)
        hidden_array_eval_list.append(en_hi[:, :].detach().cpu().numpy())
    label_eval = np.hstack(label_eval_list)
    hidden_array_eval = np.vstack(hidden_array_eval_list)

    return hidden_array_train, hidden_array_eval, label_train, label_eval

def knn(data_train, data_test, label_train, label_test, nn=9):
    label_train = np.asarray(label_train)
    label_test = np.asarray(label_test)
    print("Number of KNN Neighbours = ", nn)
    print("training feature and labels", data_train.shape, len(label_train))
    print("test feature and labels", data_test.shape, len(label_test))

    Xtr_Norm = preprocessing.normalize(data_train)
    Xte_Norm = preprocessing.normalize(data_test)

    knn = KNeighborsClassifier(n_neighbors=nn,
                               metric='cosine')  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
    knn.fit(Xtr_Norm, label_train)
    pred = knn.predict(Xte_Norm)
    acc = accuracy_score(pred, label_test)

    return acc

def clustering_knn_acc(model, train_loader, eval_loader, num_epoches=400, middle_size=125, knn_neighbours=1):
    hi_train, hi_eval, label_train, label_eval = test_extract_hidden(model, train_loader, eval_loader)

    knn_acc_1 = knn(hi_train, hi_eval, label_train, label_eval, nn=knn_neighbours)

    return knn_acc_1

def main_worker(args):
    if args.local_rank != -1:
        init_seeds(args.seed + args.local_rank)
    else:
        init_seeds(args.seed)

    # pretraining dataset and protocol
    from options import options_pretraining as options
    from options import options_retrieval as options_ret 
    if args.pre_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
        opts_ft = options_ret.opts_ntu_60_cross_view()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject(args)
        opts_ft = options_ret.opts_ntu_60_cross_subject()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
        opts_ft = options_ret.opts_ntu_120_cross_setup()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()
        opts_ft = options_ret.opts_ntu_120_cross_subject()
    elif args.pre_dataset == 'pku_v2' and args.protocol == 'cross_view':
        opts = options.opts_pku_v2_cross_view()
        opts_ft = options_ret.opts_pku_v2_cross_view()
    elif args.pre_dataset == 'pku_v2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_v2_cross_subject()
        opts_ft = options_ret.opts_pku_v2_cross_subject()

    opts.train_feeder_args['input_representation'] = args.skeleton_representation
    opts_ft.test_feeder_args['input_representation'] = args.skeleton_representation
    opts_ft.train_feeder_args['input_representation'] = args.skeleton_representation

    # create model
    print("=> creating model")

    if args.method == 'baseline_single_modality':
        model = moco.builder_single_modality_baseline.MoCo(args.skeleton_representation, opts.bi_gru_model_args,
                                    args.contrast_dim, args.contrast_k, args.contrast_m, args.contrast_t,
                                    args.teacher_t, args.student_t, args.cmd_weight, args.topk, args.mlp, modality_to_use=args.modality_to_use)
    elif args.method == 'single_modality_halp':
                        model = moco.builder_single_modality_halp.MoCo(args.skeleton_representation, opts.bi_gru_model_args,
                                                    args.contrast_dim, args.contrast_k, args.contrast_m, args.contrast_t,
                                                    args.teacher_t, args.student_t, args.cmd_weight, args.topk, args.mlp, modality_to_use=args.modality_to_use,args=args)                                       
    print("options",opts.train_feeder_args)
    print(model)

    model.cuda()

    if args.local_rank != -1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.distributed.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        print('Distributed data parallel model used')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ## Data loading code
    train_dataset = get_pretraining_set_intra(opts)
    train_dataset_ft = get_finetune_training_set(opts_ft)
    val_dataset_ft = get_finetune_validation_set(opts_ft)

    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    def worker_init_fn(worker_id):
        return np.random.seed(torch.initial_seed()%(2**31) + worker_id)  # for single gpu
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers,
        worker_init_fn=worker_init_fn, pin_memory=True, sampler=train_sampler, drop_last=True)
    train_loader_ft = torch.utils.data.DataLoader(
        train_dataset_ft, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    val_loader_ft = torch.utils.data.DataLoader(
        val_dataset_ft,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    global total_sk_seen
    total_sk_seen = 0
    queue_warmup = True
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        
        queue_warmup = True if model.total_sk_seen < 2*args.contrast_k + 1 else False

        # train for one epoch
        if args.method in ['baseline','baseline_single_modality','single_modality_halp']:
            loss_joint, loss_motion, loss_bone, top1_joint, top1_motion, top1_bone, loss_pn1, losses_moco = train(train_loader, model, criterion, optimizer, epoch, args, queue_warmup)
            if args.local_rank in [-1, 0]:
                if epoch % args.save_every == 0 or epoch == args.epochs-1 :
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        }, is_best=False, filename=args.checkpoint_path+'/checkpoint_{:04d}.pth.tar'.format(epoch))
                data_to_log = {
                    'loss_joint':loss_joint.avg,
                    'loss_motion':loss_motion.avg,
                    'loss_bone':loss_bone.avg,
                    'top1_joint':top1_joint.avg,
                    'top1_motion':top1_motion.avg,
                    'top1_bone':top1_bone.avg,
                    'loss_pn1':loss_pn1.avg,
                    'loss_moco':losses_moco.avg,
                    'epoch':epoch
                }

        if epoch%args.knn_every == 0 and not args.skip_knn:
            knn_acc = clustering_knn_acc(model, train_loader_ft, val_loader_ft, knn_neighbours=1)
            data_to_log['knn_acc'] = knn_acc

        wandb.log(data_to_log)

def train(train_loader, model, criterion, optimizer, epoch, args, queue_warmup):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    losses_joint = AverageMeter('Loss Joint', ':6.3f')
    losses_motion = AverageMeter('Loss Motion', ':6.3f')
    losses_bone = AverageMeter('Loss Bone', ':6.3f')
    top1_joint = AverageMeter('Acc Joint@1', ':6.2f')
    top1_motion = AverageMeter('Acc Motion@1', ':6.2f')
    top1_bone = AverageMeter('Acc Bone@1', ':6.2f')

    losses_pn1 = AverageMeter('Loss PN-1', ':6.3f')
    losses_moco = AverageMeter('Loss MoCo', ':6.3f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses_joint, losses_motion, losses_bone, top1_joint, top1_motion, top1_bone],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch,optimizer.param_groups[0]['lr']))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_v1, input_v2) in tqdm(enumerate(train_loader),total=len(train_loader)):
        inputs= [input_v1,input_v2]
        inputs[0] =inputs[0].float().cuda(non_blocking=True)
        inputs[1] =inputs[1].float().cuda(non_blocking=True)

        total_sk_seen = model.total_sk_seen

        if i%args.update_prototypes_every == 0 and queue_warmup == False and args.method != 'baseline':
            update_prototypes = True
        else:
            update_prototypes = False

        if (total_sk_seen >= 2*args.contrast_k + 1 and queue_warmup == True and args.method != 'baseline'):
            update_prototypes = True
            cprint('First update to the prototypes','red')
            model.update_prototypes(True)
            queue_warmup = False

        generate_this_step = False if queue_warmup else True

        # compute output
        if args.method in ['baseline_single_modality']:
            output, target, pn_loss, _, _ = model(inputs[0], inputs[1])
            batch_size = output.size(0)
            loss = criterion(output, target)
            if args.modality_to_use == 'joint':
                acc1_joint, _ = accuracy(output, target, topk=(1, 5))
                top1_joint.update(acc1_joint[0], batch_size)
            elif args.modality_to_use == 'motion':
                acc1_motion, _ = accuracy(output, target, topk=(1, 5))
                top1_motion.update(acc1_motion[0], batch_size)
            elif args.modality_to_use == 'bone':
                acc1_bone, _ = accuracy(output, target, topk=(1, 5))
                top1_bone.update(acc1_bone[0], batch_size)
    
        elif args.method in ['single_modality_halp']:
            output, target, pn_loss = model(inputs[0], inputs[1], update_prototypes=update_prototypes,
                                                                                 generate_this_step=generate_this_step)
            batch_size = output.size(0)
            loss = criterion(output, target) + pn_loss
            # measure accuracy of model m1 and m2 individually
            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            if args.modality_to_use == 'joint':
                acc1_joint, _ = accuracy(output, target, topk=(1, 5))
                top1_joint.update(acc1_joint[0], batch_size)
            elif args.modality_to_use == 'motion':
                acc1_motion, _ = accuracy(output, target, topk=(1, 5))
                top1_motion.update(acc1_motion[0], batch_size)
            elif args.modality_to_use == 'bone':
                acc1_bone, _ = accuracy(output, target, topk=(1, 5))
                top1_bone.update(acc1_bone[0], batch_size)
            losses_pn1.update(pn_loss,batch_size)

        losses.update(loss.item(), batch_size)
        if args.modality_to_use == 'joint':
            losses_joint.update(loss.item(), batch_size)
        elif args.modality_to_use == 'motion':
            losses_motion.update(loss.item(), batch_size)
        elif args.modality_to_use == 'bone':
            losses_bone.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        model.total_sk_seen += args.batch_size

    return losses_joint, losses_motion, losses_bone, top1_joint, top1_motion, top1_bone, losses_pn1, losses_moco

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    Path(filename).parent.mkdir(parents=True,exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
