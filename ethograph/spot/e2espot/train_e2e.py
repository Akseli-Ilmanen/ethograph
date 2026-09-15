#!/usr/bin/env python3
# Vendored from jhong93/spot (BSD-3-Clause) — see NOTICE.md
# ruff: noqa
""" Training for E2E-Spot """

import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
from tabulate import tabulate
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from torch.utils.data import DataLoader
import torchvision
import timm
from tqdm import tqdm

from ethograph.spot.e2espot.model.common import step, BaseRGBModel
from ethograph.spot.e2espot.model.shift import make_temporal_shift
from ethograph.spot.e2espot.model.modules import *
from ethograph.spot.e2espot.dataset.frame import ActionSpotDataset, ActionSpotVideoDataset
from ethograph.spot.e2espot.util.eval import process_frame_predictions
from ethograph.spot.e2espot.util.io import load_json, store_json, store_gz_json, clear_files
from ethograph.spot.e2espot.util.dataset import DATASETS, load_classes
from ethograph.spot.e2espot.util.score import compute_mAPs

EPOCH_NUM_FRAMES = 500000

BASE_NUM_WORKERS = 4

BASE_NUM_VAL_EPOCHS = 20

INFERENCE_BATCH_SIZE = 4

# Peak memory tracks frames per batch, not clips, so a batch of 4 clips means
# 4 x clip_len frames — 800 at clip_len 200, which pages a 10 GB card. Size
# the inference batch by frames instead; upstream's 4 x 100 is the ceiling.
INFERENCE_BATCH_FRAMES = 400


def inference_batch_size(clip_len):
    return max(1, INFERENCE_BATCH_FRAMES // clip_len)


# Prevent the GRU params from going too big (cap it at a RegNet-Y 800MF)
MAX_GRU_HIDDEN_DIM = 768


def get_args():
    parser = argparse.ArgumentParser()
    # ethograph: a dataset is a name under data/ (upstream) or a directory
    # anywhere (`os.path.join('data', <absolute>)` is the absolute path)
    parser.add_argument('dataset', type=dataset_name_or_dir)
    parser.add_argument('frame_dir', type=str, help='Path to extracted frames')

    parser.add_argument('--modality', type=str, choices=['rgb', 'bw', 'flow'],
                        default='rgb')
    parser.add_argument(
        '-m', '--feature_arch', type=str, required=True, choices=[
            # From torchvision
            'rn18',
            'rn18_tsm',
            'rn18_gsm',
            'rn50',
            'rn50_tsm',
            'rn50_gsm',

            # From timm (following its naming conventions)
            'rny002',
            'rny002_tsm',
            'rny002_gsm',
            'rny008',
            'rny008_tsm',
            'rny008_gsm',
            'rny008_msagsm',
            'rny002_msagsm',
            'rn18_msagsm',

            # From timm
            'convnextt',
            'convnextt_tsm',
            'convnextt_gsm'
        ], help='CNN architecture for feature extraction')
    parser.add_argument(
        '-t', '--temporal_arch', type=str, default='gru',
        choices=['', 'gru', 'deeper_gru', 'mstcn', 'asformer'],
        help='Spotting architecture, after spatial pooling')

    parser.add_argument('--clip_len', type=int, default=100)
    parser.add_argument('--stride', type=int, default=1,
                        help='Read every k-th frame: k times the context per clip for the same compute, at k-frame resolution')
    parser.add_argument('--epoch_num_frames', type=int, default=EPOCH_NUM_FRAMES,
                        help='Frames per epoch, independent of dataset size. '
                             'Halving it halves the wall clock of an epoch')
    parser.add_argument('--crop_dim', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')

    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-s', '--save_dir', type=str, required=True,
                        help='Dir to save checkpoints and predictions')

    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint in <save_dir>')

    parser.add_argument('--start_val_epoch', type=int)
    parser.add_argument('--criterion', choices=['map', 'loss'], default='map')

    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3],
                        help='ethograph: 1 = labels (upstream), 2 = distil a '
                             'teacher embedding (no labels), 3 = labels with '
                             'the CNN frozen')
    parser.add_argument('--teacher_dir', type=str, default=None,
                        help='stage 2: folder of per-video teacher '
                             'embeddings ({video}.npz)')
    parser.add_argument('--fuse_dir', type=str, default=None,
                        help='ethograph: folder of per-video pose blocks '
                             '({video}.npz, features (T\', D)) concatenated to '
                             'the CNN features before the temporal head')
    parser.add_argument('--fuse_dim', type=int, default=None,
                        help='ethograph: width D of the pose block')
    parser.add_argument('--fuse_dropout', type=float, default=0.0,
                        help='ethograph: share of training clips whose pose '
                             'block is zeroed, so the pixels carry the event '
                             'on their own too')
    parser.add_argument('--distil_dim', type=int, default=None,
                        help='stage 2/3: width of the teacher embedding the '
                             'student projects onto')
    parser.add_argument('--init_from', type=str, default=None,
                        help='checkpoint to start the weights from')
    parser.add_argument('--shift_dilations', type=str, default='1,2,3',
                        help='MSAGSM only: comma-separated reach of each '
                             'gated-shift branch, in (strided) frames')
    parser.add_argument('--attention_groups', type=int, default=2,
                        help='MSAGSM only: channel groups of the spatial '
                             'attention')
    parser.add_argument('--dilate_len', type=int, default=0,
                        help='Label dilation when training')
    parser.add_argument('--mixup', type=bool, default=True)

    parser.add_argument('-j', '--num_workers', type=int,
                        help='Base number of dataloader workers')

    # Sample based on foreground
    parser.add_argument('--fg_upsample', type=float)

    parser.add_argument('-mgpu', '--gpu_parallel', action='store_true')
    return parser.parse_args()


class E2EModel(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, num_classes, feature_arch, temporal_arch, clip_len,
                     modality, shift_dilations=None, attention_groups=2,
                     distil_dim=None, fuse_dim=None):
            super().__init__()
            self._distil_dim = distil_dim
            self._fuse_dim = fuse_dim or 0
            is_rgb = modality == 'rgb'
            in_channels = {'flow': 2, 'bw': 1, 'rgb': 3}[modality]

            if feature_arch.startswith(('rn18', 'rn50')):
                resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
                features = getattr(
                    torchvision.models, resnet_name)(pretrained=is_rgb)
                feat_dim = features.fc.in_features
                features.fc = nn.Identity()
                # import torchsummary
                # print(torchsummary.summary(features.to('cuda'), (3, 224, 224)))

                # Flow has only two input channels
                if not is_rgb:
                    #FIXME: args maybe wrong for larger resnet
                    features.conv1 = nn.Conv2d(
                        in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                        padding=(3, 3), bias=False)

            elif feature_arch.startswith(('rny002', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[feature_arch.rsplit('_', 1)[0]], pretrained=is_rgb)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

                if not is_rgb:
                    features.stem.conv = nn.Conv2d(
                        in_channels, 32, kernel_size=(3, 3), stride=(2, 2),
                        padding=(1, 1), bias=False)

            elif 'convnextt' in feature_arch:
                features = timm.create_model('convnext_tiny', pretrained=is_rgb)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

                if not is_rgb:
                    features.stem[0] = nn.Conv2d(
                        in_channels, 96, kernel_size=4, stride=4)

            else:
                raise NotImplementedError(feature_arch)

            # Add Temporal Shift Modules
            self._require_clip_len = -1
            if feature_arch.endswith('_tsm'):
                make_temporal_shift(features, clip_len, is_gsm=False)
                self._require_clip_len = clip_len
            elif feature_arch.endswith('_gsm'):
                make_temporal_shift(features, clip_len, is_gsm=True)
                self._require_clip_len = clip_len
            elif feature_arch.endswith('_msagsm'):
                # ethograph's multi-scale gated shift in place of _GSM
                from functools import partial
                from ethograph.spot.msagsm import MultiScaleGatedShift
                make_temporal_shift(
                    features, clip_len, is_gsm=True,
                    shift_module=partial(
                        MultiScaleGatedShift,
                        dilations=tuple(shift_dilations or (1, 2, 3)),
                        attention_groups=attention_groups))
                self._require_clip_len = clip_len

            self._features = features
            self._feat_dim = feat_dim
            # ethograph: the temporal head reads the CNN features plus the
            # pose block, when there is one
            feat_dim = feat_dim + self._fuse_dim

            if 'gru' in temporal_arch:
                hidden_dim = feat_dim
                if hidden_dim > MAX_GRU_HIDDEN_DIM:
                    hidden_dim = MAX_GRU_HIDDEN_DIM
                    print('Clamped GRU hidden dim: {} -> {}'.format(
                        feat_dim, hidden_dim))
                if temporal_arch in ('gru', 'deeper_gru'):
                    self._pred_fine = GRUPrediction(
                        feat_dim, num_classes, hidden_dim,
                        num_layers=3 if temporal_arch[0] == 'd' else 1)
                else:
                    raise NotImplementedError(temporal_arch)
            elif temporal_arch == 'mstcn':
                self._pred_fine = TCNPrediction(feat_dim, num_classes, 3)
            elif temporal_arch == 'asformer':
                self._pred_fine = ASFormerPrediction(feat_dim, num_classes, 3)
            elif temporal_arch == '':
                self._pred_fine = FCPrediction(feat_dim, num_classes)
            else:
                raise NotImplementedError(temporal_arch)

            # ethograph: the student's per-frame embedding (the GRU output)
            # projected onto the teacher's width, for distillation
            if distil_dim is not None:
                if 'gru' not in temporal_arch:
                    raise NotImplementedError(
                        'distillation needs a GRU head, got ' + temporal_arch)
                self._distil_proj = nn.Linear(
                    2 * self._pred_fine._gru.hidden_size, distil_dim)

        def forward(self, x, fuse=None, return_embedding=False):
            batch_size, true_clip_len, channels, height, width = x.shape

            clip_len = true_clip_len
            if self._require_clip_len > 0:
                # TSM module requires clip len to be known
                assert true_clip_len <= self._require_clip_len, \
                    'Expected {}, got {}'.format(
                        self._require_clip_len, true_clip_len)
                if true_clip_len < self._require_clip_len:
                    x = F.pad(
                        x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                    clip_len = self._require_clip_len

            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._feat_dim)

            if true_clip_len != clip_len:
                # Undo padding
                im_feat = im_feat[:, :true_clip_len, :]

            if self._fuse_dim:
                # ethograph: the pose block rides beside the CNN features
                if fuse is None:
                    raise ValueError(
                        'this model was trained with a pose block of width '
                        '{} and got none'.format(self._fuse_dim))
                im_feat = torch.cat(
                    [im_feat, fuse[:, :true_clip_len, :].to(im_feat.dtype)],
                    dim=-1)

            if return_embedding:
                # ethograph: GRU output before the classifier, plus its
                # projection onto the teacher's width
                y, _ = self._pred_fine._gru(im_feat)
                pred = self._pred_fine._fc_out(self._pred_fine._dropout(y))
                return pred, self._distil_proj(y)
            return self._pred_fine(im_feat)

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))
            print('  CNN features:',
                sum(p.numel() for p in self._features.parameters()))
            print('  Temporal:',
                sum(p.numel() for p in self._pred_fine.parameters()))

    def __init__(self, num_classes, feature_arch, temporal_arch, clip_len,
                 modality, device='cuda', multi_gpu=False,
                 shift_dilations=None, attention_groups=2,
                 distil_dim=None, stage=1, fuse_dim=None, fuse_dropout=0.0):
        self.device = device
        self._multi_gpu = multi_gpu
        self._stage = stage
        self._fuse_dropout = fuse_dropout
        self._model = E2EModel.Impl(
            num_classes, feature_arch, temporal_arch, clip_len, modality,
            shift_dilations=shift_dilations,
            attention_groups=attention_groups,
            distil_dim=distil_dim, fuse_dim=fuse_dim)
        self._model.print_stats()
        if stage == 3:
            # ethograph: the head step trains the GRU + classifier only
            for param in self._model._features.parameters():
                param.requires_grad = False
            print('=> Stage 3: CNN frozen, {} trainable params'.format(
                sum(p.numel() for p in self._model.parameters()
                    if p.requires_grad)))

        if multi_gpu:
            self._model = nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
              acc_grad_iter=1, fg_weight=5):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = loader.dataset.load_frame_gpu(batch, self.device)
                label = batch['label'].to(self.device)
                fuse = batch.get('fuse')
                if fuse is not None:
                    fuse = fuse.to(self.device)
                    if optimizer is not None and self._fuse_dropout > 0:
                        # ethograph: modality dropout -- a share of clips see
                        # no pose, so the pixels are trained to carry the
                        # event on their own too
                        keep = (torch.rand(fuse.shape[0], 1, 1,
                                           device=fuse.device)
                                >= self._fuse_dropout).to(fuse.dtype)
                        fuse = fuse * keep

                # Depends on whether mixup is used
                label = label.flatten() if len(label.shape) == 2 \
                    else label.view(-1, label.shape[-1])

                with torch.cuda.amp.autocast():
                    if self._stage == 2:
                        # ethograph: match the teacher's embedding, frame by
                        # frame, on the frames the teacher has (not padding)
                        _, emb = self._model(
                            frame, fuse=fuse, return_embedding=True)
                        target = batch['embedding'].to(self.device)
                        mask = batch['embedding_mask'].to(self.device)
                        if mask.any():
                            loss = F.mse_loss(emb[mask].float(),
                                              target[mask].float())
                        else:
                            loss = emb.sum() * 0.
                    else:
                        pred = self._model(frame, fuse=fuse)

                        loss = 0.
                        if len(pred.shape) == 3:
                            pred = pred.unsqueeze(0)

                        for i in range(pred.shape[0]):
                            loss += F.cross_entropy(
                                pred[i].reshape(-1, self._num_classes), label,
                                **ce_kwargs)

                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter,
                         lr_scheduler=lr_scheduler,
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq, use_amp=True, fuse=None):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        if fuse is not None:
            # ethograph: one clip's pose block, or a batch of them; a clip
            # augmented into several views shares its block
            if not isinstance(fuse, torch.Tensor):
                fuse = torch.FloatTensor(fuse)
            if len(fuse.shape) == 2:
                fuse = fuse.unsqueeze(0)
            if fuse.shape[0] == 1 and seq.shape[0] > 1:
                fuse = fuse.expand(seq.shape[0], -1, -1)
            fuse = fuse.to(self.device)

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                pred = self._model(seq, fuse=fuse)
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = torch.softmax(pred, axis=2)
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()


def evaluate(model, dataset, split, classes, save_pred, calc_stats=True,
             save_scores=True):
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))

    # Do not up the batch size if the dataset augments
    batch_size = 1 if dataset.augment else inference_batch_size(dataset.clip_len)

    for clip in tqdm(DataLoader(
            dataset, num_workers=BASE_NUM_WORKERS * 2, pin_memory=True,
            batch_size=batch_size
    )):
        if batch_size > 1:
            # Batched by dataloader
            _, batch_pred_scores = model.predict(
                clip['frame'], fuse=clip.get('fuse'))

            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip['start'][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:end - start, :]
                scores[start:end, :] += pred_scores
                support[start:end] += 1

        else:
            # Batched by dataset
            scores, support = pred_dict[clip['video'][0]]

            start = clip['start'][0].item()
            fuse = clip['fuse'][0] if 'fuse' in clip else None
            _, pred_scores = model.predict(clip['frame'][0], fuse=fuse)
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:,:end - start, :]

            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]

    err, f1, pred_events, pred_events_high_recall, pred_scores = \
        process_frame_predictions(dataset, classes, pred_dict)

    avg_mAP = None
    if calc_stats:
        print('=== Results on {} (w/o NMS) ==='.format(split))
        print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

        def get_f1_tab_row(str_k):
            k = classes[str_k] if str_k != 'any' else None
            return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
        rows = [get_f1_tab_row('any')]
        for c in sorted(classes):
            rows.append(get_f1_tab_row(c))
        print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                       floatfmt='0.2f'))
        print()

        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall)
        avg_mAP = np.mean(mAPs[1:])

    if save_pred is not None:
        store_json(save_pred + '.json', pred_events)
        store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)
        if save_scores:
            store_gz_json(save_pred + '.score.json.gz', pred_scores)
    return avg_mAP


def get_last_epoch(save_dir):
    max_epoch = -1
    for file_name in os.listdir(save_dir):
        if not file_name.startswith('optim_'):
            continue
        epoch = int(os.path.splitext(file_name)[0].split('optim_')[1])
        if epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def get_best_epoch_and_history(save_dir, criterion):
    data = load_json(os.path.join(save_dir, 'loss.json'))
    if criterion == 'map':
        key = 'val_mAP'
        best = max(data, key=lambda x: x[key])
    else:
        key = 'val'
        best = min(data, key=lambda x: x[key])
    return data, best['epoch'], best[key]


def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    dataset_kwargs = {
        'crop_dim': args.crop_dim, 'dilate_len': args.dilate_len,
        'mixup': args.mixup, 'stride': args.stride,
        'teacher_dir': args.teacher_dir if args.stage == 2 else None,
        'fuse_dir': args.fuse_dir
    }
    if (args.fuse_dir is None) != (args.fuse_dim is None):
        raise ValueError('--fuse_dir and --fuse_dim go together')
    if args.stage == 2 and args.teacher_dir is None:
        raise ValueError('--stage 2 needs --teacher_dir')
    if args.stage == 2:
        # matching embeddings, not labels: mixup would blend two clips'
        # frames against one clip's targets (and `--mixup False` parses as
        # True through argparse's bool, so it is forced off here)
        dataset_kwargs['mixup'] = False

    if args.fg_upsample is not None:
        assert args.fg_upsample > 0
        dataset_kwargs['fg_upsample'] = args.fg_upsample

    print('Dataset size:', dataset_len)
    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.modality, args.clip_len, dataset_len,
        is_eval=False, **dataset_kwargs)
    train_data.print_info()
    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.modality, args.clip_len, dataset_len // 4,
        **dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    if args.criterion == 'map':
        # Only perform mAP evaluation during training if criterion is mAP
        val_data_frames = ActionSpotVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.modality, args.clip_len,
            crop_dim=args.crop_dim, overlap_len=0, stride=args.stride,
            fuse_dir=args.fuse_dir)

    return classes, train_data, val_data, val_data_frames


def load_from_save(
        args, model, optimizer, scaler, lr_scheduler
):
    assert args.save_dir is not None
    epoch = get_last_epoch(args.save_dir)

    print('Loading from epoch {}'.format(epoch))
    model.load(torch.load(os.path.join(
        args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))))

    if args.resume:
        # print('(Resume) Train loss:', model.epoch(train_loader))
        # print('(Resume) Val loss:', model.epoch(val_loader))
        opt_data = torch.load(os.path.join(
            args.save_dir, 'optim_{:03d}.pt'.format(epoch)))
        optimizer.load_state_dict(opt_data['optimizer_state_dict'])
        scaler.load_state_dict(opt_data['scaler_state_dict'])
        lr_scheduler.load_state_dict(opt_data['lr_state_dict'])

    losses, best_epoch, best_criterion = get_best_epoch_and_history(
        args.save_dir, args.criterion)
    return epoch, losses, best_epoch, best_criterion


def dataset_name_or_dir(text):
    if text in DATASETS or os.path.isdir(text):
        return text
    raise argparse.ArgumentTypeError(
        '{} is neither a dataset under data/ ({}) nor a directory'.format(
            text, ', '.join(DATASETS)))


def parse_shift_dilations(text):
    return [int(d) for d in str(text).split(',') if d.strip()]


def store_config(file_path, args, num_epochs, classes):
    config = {
        'dataset': args.dataset,
        'num_classes': len(classes),
        'modality': args.modality,
        'feature_arch': args.feature_arch,
        'temporal_arch': args.temporal_arch,
        'clip_len': args.clip_len,
        'stride': args.stride,
        'batch_size': args.batch_size,
        'crop_dim': args.crop_dim,
        'num_epochs': num_epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.start_val_epoch,
        'gpu_parallel': args.gpu_parallel,
        'epoch_num_frames': args.epoch_num_frames,
        'dilate_len': args.dilate_len,
        'mixup': args.mixup,
        'fg_upsample': args.fg_upsample,
        'shift_dilations': parse_shift_dilations(args.shift_dilations),
        'attention_groups': args.attention_groups,
        'stage': args.stage,
        'distil_dim': args.distil_dim,
        'teacher_dir': args.teacher_dir,
        'init_from': args.init_from,
        'fuse_dir': args.fuse_dir,
        'fuse_dim': args.fuse_dim,
        'fuse_dropout': args.fuse_dropout
    }
    if file_path is None:
        import json
        print(json.dumps(config, indent=2, sort_keys=True))
    else:
        store_json(file_path, config, pretty=True)


class _WorkerSeed:
    """Picklable worker_init_fn: spawned workers (Windows) cannot receive a
    closure over main()'s locals, so the epoch is an attribute set per epoch
    and pickled with the object when each worker starts."""

    def __init__(self):
        self.epoch = 0

    def __call__(self, id):
        random.seed(id + self.epoch * 100)


def get_num_train_workers(args):
    n = BASE_NUM_WORKERS * 2
    # if args.gpu_parallel:
    #     n *= torch.cuda.device_count()
    return min(os.cpu_count(), n)


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    if args.num_workers is not None:
        global BASE_NUM_WORKERS
        BASE_NUM_WORKERS = args.num_workers

    assert args.batch_size % args.acc_grad_iter == 0
    if args.start_val_epoch is None:
        args.start_val_epoch = args.num_epochs - BASE_NUM_VAL_EPOCHS
    if args.crop_dim <= 0:
        args.crop_dim = None

    classes, train_data, val_data, val_data_frames = get_datasets(args)

    worker_init_fn = _WorkerSeed()
    loader_batch_size = args.batch_size // args.acc_grad_iter
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=get_num_train_workers(args),
        prefetch_factor=1, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=BASE_NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    model = E2EModel(
        len(classes) + 1, args.feature_arch, args.temporal_arch,
        clip_len=args.clip_len, modality=args.modality,
        multi_gpu=args.gpu_parallel,
        shift_dilations=parse_shift_dilations(args.shift_dilations),
        attention_groups=args.attention_groups,
        distil_dim=args.distil_dim, stage=args.stage,
        fuse_dim=args.fuse_dim, fuse_dropout=args.fuse_dropout)
    if args.init_from is not None and not args.resume:
        # ethograph: warm start (a baseline has no projection layer yet, a
        # stage-2 checkpoint has one -- both load)
        print('=> Initialising weights from', args.init_from)
        model.load(torch.load(args.init_from), strict=False)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    # Warmup schedule
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    num_epochs, lr_scheduler = get_lr_scheduler(
        args, optimizer, num_steps_per_epoch)

    losses = []
    best_epoch = None
    best_criterion = 0 if args.criterion == 'map' else float('inf')

    epoch = 0
    if args.resume:
        epoch, losses, best_epoch, best_criterion = load_from_save(
            args, model, optimizer, scaler, lr_scheduler)
        epoch += 1

    # Write it to console
    store_config(None, args, num_epochs, classes)

    for epoch in range(epoch, num_epochs):
        worker_init_fn.epoch = epoch
        train_loss = model.epoch(
            train_loader, optimizer, scaler,
            lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter)
        val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter)
        print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
            epoch, train_loss, val_loss))

        val_mAP = 0
        if args.criterion == 'loss':
            if val_loss < best_criterion:
                best_criterion = val_loss
                best_epoch = epoch
                print('New best epoch!')
        elif args.criterion == 'map':
            if epoch >= args.start_val_epoch:
                pred_file = None
                if args.save_dir is not None:
                    pred_file = os.path.join(
                        args.save_dir, 'pred-val.{}'.format(epoch))
                    os.makedirs(args.save_dir, exist_ok=True)
                val_mAP = evaluate(model, val_data_frames, 'VAL', classes,
                                    pred_file, save_scores=False)
                if args.criterion == 'map' and val_mAP > best_criterion:
                    best_criterion = val_mAP
                    best_epoch = epoch
                    print('New best epoch!')
        else:
            print('Unknown criterion:', args.criterion)

        losses.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss,
            'val_mAP': val_mAP})
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            store_json(os.path.join(args.save_dir, 'loss.json'), losses,
                        pretty=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir,
                    'checkpoint_{:03d}.pt'.format(epoch)))
            clear_files(args.save_dir, r'optim_\d+\.pt')
            torch.save(
                {'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'lr_state_dict': lr_scheduler.state_dict()},
                os.path.join(args.save_dir,
                                'optim_{:03d}.pt'.format(epoch)))
            store_config(os.path.join(args.save_dir, 'config.json'),
                            args, num_epochs, classes)

    print('Best epoch: {}\n'.format(best_epoch))

    if args.save_dir is not None:
        model.load(torch.load(os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

        # Evaluate on VAL if not already done
        eval_splits = ['val'] if args.criterion != 'map' else []

        # Evaluate on hold out splits
        eval_splits += ['test', 'challenge']
        for split in eval_splits:
            split_path = os.path.join(
                'data', args.dataset, '{}.json'.format(split))
            if os.path.exists(split_path):
                split_data = ActionSpotVideoDataset(
                    classes, split_path, args.frame_dir, args.modality,
                    args.clip_len, overlap_len=args.clip_len // 2,
                    stride=args.stride,
                    crop_dim=args.crop_dim,
                    fuse_dir=args.fuse_dir)  # ethograph: the hold-out splits read the pose block too
                split_data.print_info()

                pred_file = None
                if args.save_dir is not None:
                    pred_file = os.path.join(
                        args.save_dir, 'pred-{}.{}'.format(split, best_epoch))

                evaluate(model, split_data, split.upper(), classes, pred_file,
                         calc_stats=split != 'challenge')


if __name__ == '__main__':
    main(get_args())
