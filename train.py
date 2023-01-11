from models.arguments import parse_args

import os
import errno
import copy

from common.camera import world_to_camera
from common.loss import *
from common.generators_pspt import PoseGenerator
from common.function import *
from time import time
from common.utils import deterministic_random
from torch.utils.data import DataLoader

from common.prefetcher import Prefetcher_train, Prefetcher_test
from models.utils import ReDirectSTD, parse_param
from models.mhr_net import MHR_Net

args = parse_args()

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

ReDirectSTD(os.path.join(args.checkpoint, 'log.txt'), 'stdout', False)
print(args)

# set the seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# set cudnn to reproducibility mode
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset

    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset

    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        if args.dataset != 'gt' and action == 'Directions' and subject == 'S11':
            continue
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
            action,
            subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            cam = dataset.cameras()[subject][cam_idx]
            kps -= kps[:, :1]
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',')

test_actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
                'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
action_frames = {}

def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True, valid_info=False):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if valid_info:
                action_st = np.concatenate(out_poses_2d).shape[0] if len(out_poses_2d) > 0 else 0

            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for i, cam in enumerate(cams):
                    if 'intrinsic' in cam:
                        out_camera_params.append(np.tile((cam['intrinsic'])[None, :], (len(poses_2d[i]), 1)))

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

            if valid_info:
                action_name = action
                if action_name[-2] == ' ':
                    action_name = action_name[:-2]
                # recording the start/end frame of current action, also note that one action can appear in different subjects
                if not action_name in action_frames:
                    action_frames[action_name] = [(action_st, np.concatenate(out_poses_2d).shape[0])]
                else:
                    action_frames[action_name].append((action_st, np.concatenate(out_poses_2d).shape[0]))

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
                out_camera_params[i] = out_camera_params[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter, valid_info=True)
# print(len(poses_valid))
print(action_frames)

hparams = parse_param(args.hparams)

model_pos_train = MHR_Net(hparams)
model_pos = MHR_Net(hparams)

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, cameras_valid),
                          batch_size=1024, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)


lr = args.learning_rate

optimizer = torch.optim.SGD(model_pos_train.parameters(), lr=lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
# optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

lr_decay = args.lr_decay

losses_3d_train_rp = []
losses_3d_train_cs = []
losses_3d_train_pc = []
losses_3d_train_eval = []
errors_3d_valid_p1 = []
errors_3d_valid_p2 = []

epoch = 0

train_loader = DataLoader(PoseGenerator(poses_train, poses_train_2d, cameras_train, training=True), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers, pin_memory=True)

if args.resume:
    epoch = checkpoint['epoch']
    if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

    lr = checkpoint['lr']

best_result_p1 = 999
best_result_p2 = 999
best_result_errors_3d_valid_p1_action = []
best_result_errors_3d_valid_p2_action = []

print('*** Start training ***')
while epoch < args.epochs:
    start_time = time()
    epoch_loss_3d_train_rp = 0
    epoch_loss_3d_train_cs = 0
    epoch_loss_3d_train_pc = 0
    N = 0
    model_pos_train.train()

    prefetcher_train = Prefetcher_train(train_loader)
    inputs_2d = prefetcher_train.next()
    i = 0

    while inputs_2d is not None:
        optimizer.zero_grad()
        if epoch + 1 <= 9:
            use_procrust = False
        else:
            use_procrust = True
        if epoch < 15:
            preds = model_pos_train(inputs_2d, align_to_root=True, use_procrust=use_procrust)
        else:
            preds = model_pos_train(inputs_2d, align_to_root=False, use_procrust=use_procrust)

        loss_reprojection = preds['l_reprojection']
        loss_consistancy = preds['l_cycle_consistent']
        if use_procrust:
            loss_procrust = preds['l_procrust']
        else:
            loss_procrust = None

        loss_total = args.reploss_weight * loss_reprojection + loss_consistancy
        if use_procrust:
            loss_total += loss_procrust
        epoch_loss_3d_train_rp += inputs_2d.shape[0] * loss_reprojection.item()
        epoch_loss_3d_train_cs += inputs_2d.shape[0] * loss_consistancy.item()
        if use_procrust:
            if loss_procrust > 0:
                epoch_loss_3d_train_pc += inputs_2d.shape[0] * loss_procrust.item()

        N += inputs_2d.shape[0]

        loss_total.backward()

        optimizer.step()

        i += 1
        inputs_2d = prefetcher_train.next()

    losses_3d_train_rp.append(epoch_loss_3d_train_rp / N)
    losses_3d_train_cs.append(epoch_loss_3d_train_cs / N)
    if use_procrust:
        losses_3d_train_pc.append(epoch_loss_3d_train_pc / N)

    # End-of-epoch evaluation
    with torch.no_grad():
        model_pos.load_state_dict(model_pos_train.state_dict())
        model_pos.eval()

        epoch_error_p1 = 0
        epoch_error_p2 = 0
        N = 0

        frame_dist = []
        frame_p_mpjpe = []

        if not args.no_eval:
            # Evaluate on test set
            prefetcher_test = Prefetcher_test(valid_loader)
            inputs_3d, inputs_2d, inputs_scale = prefetcher_test.next()
            i = 0

            while inputs_3d is not None:
                preds = model_pos(inputs_2d)

                shape_camera_coord = preds['shape_camera_coord']
                depth = shape_camera_coord[:, :, 2:3]
                shape_camera_coord = torch.cat((inputs_2d * (5 + depth), depth), dim=2)

                shape_camera_coord_flip = shape_camera_coord.clone()
                shape_camera_coord_flip[:, :, 2] = -shape_camera_coord[:, :, 2]
                shape_camera_coord = calibrate_by_scale(shape_camera_coord, inputs_3d)
                shape_camera_coord_flip = calibrate_by_scale(shape_camera_coord_flip, inputs_3d)

                shape_camera_coord = shape_camera_coord - shape_camera_coord[:, 0:1, :]
                shape_camera_coord_flip = shape_camera_coord_flip - shape_camera_coord_flip[:, 0:1, :]
                inputs_3d = inputs_3d - inputs_3d[:, 0:1, :]
                inputs_scale = np.asarray(inputs_scale)

                dist = calc_dist(shape_camera_coord, inputs_3d)
                p_dist = p_mpjpe(shape_camera_coord, inputs_3d)
                dist_flip = calc_dist(shape_camera_coord_flip, inputs_3d)
                p_dist_flip = p_mpjpe(shape_camera_coord_flip, inputs_3d)

                dist_best = np.minimum(dist, dist_flip)
                p_dist_best = np.minimum(p_dist, p_dist_flip)

                dist_best = dist_best * np.asarray(inputs_scale).reshape(-1)
                p_dist_best = p_dist_best * np.asarray(inputs_scale).reshape(-1)

                # save for each action
                frame_dist.append(dist_best.reshape(-1))
                frame_p_mpjpe.append(p_dist_best.reshape(-1))

                loss_3d_p1 = dist_best.mean()
                loss_3d_p2 = p_dist_best.mean()
                epoch_error_p1 += inputs_3d.shape[0] * loss_3d_p1
                epoch_error_p2 += inputs_3d.shape[0] * loss_3d_p2
                N += inputs_3d.shape[0]

                i += 1
                inputs_3d, inputs_2d, inputs_scale = prefetcher_test.next()

            errors_3d_valid_p1.append(epoch_error_p1 / N)
            errors_3d_valid_p2.append(epoch_error_p2 / N)

            # calculate metrics for each action
            frame_dist = np.concatenate(frame_dist)
            frame_p_mpjpe = np.concatenate(frame_p_mpjpe)

            errors_3d_valid_p1_action = {}
            errors_3d_valid_p2_action = {}
            for action in test_actions:
                dist_action = 0
                p_dist_action = 0
                N_action = 0
                for (st, ed) in action_frames[action]:
                    dist_action += np.sum(frame_dist[st:ed])
                    p_dist_action += np.sum(frame_p_mpjpe[st:ed])
                    N_action += ed - st
                dist_action /= N_action
                p_dist_action /= N_action
                errors_3d_valid_p1_action[action] = dist_action
                errors_3d_valid_p2_action[action] = p_dist_action

    elapsed = (time() - start_time) / 60

    if args.no_eval:
        print('[%d] time %.2f lr %f reprojection_loss %f consistent_loss %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train_rp[-1] * 1000,
            losses_3d_train_cs[-1] * 1000,))
    elif use_procrust:
        print('[%d] time %.2f lr %f reprojection_loss %f consistent_loss %f procrust_loss %f MPJPE %f P-MPJPE %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train_rp[-1] * 1000,
            losses_3d_train_cs[-1] * 1000,
            losses_3d_train_pc[-1] * 1000,
            errors_3d_valid_p1[-1] * 1000,
            errors_3d_valid_p2[-1] * 1000))
    else:
        print('[%d] time %.2f lr %f reprojection_loss %f consistent_loss %f procrust_loss not_used MPJPE %f P-MPJPE %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train_rp[-1] * 1000,
            losses_3d_train_cs[-1] * 1000,
            errors_3d_valid_p1[-1] * 1000,
            errors_3d_valid_p2[-1] * 1000))

    # Decay learning rate exponentially
    if (epoch + 1) % args.epoch_lr_decay == 0:
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
    epoch += 1

    # Save checkpoint if necessary
    if (epoch >= 1 and epoch % 3 == 0) or epoch == args.epochs:
        chk_path = os.path.join(args.checkpoint, 'model_epoch_{}.bin'.format(epoch))
        print('Saving checkpoint to', chk_path)
        torch.save({
            'epoch': epoch,
            'lr': lr,
            'optimizer': optimizer.state_dict(),
            'model_pos': model_pos_train.state_dict(),
        }, chk_path)

    if errors_3d_valid_p2[-1] < best_result_p2:
        chk_path = os.path.join(args.checkpoint, 'model_best.bin')
        print('Saving checkpoint to', chk_path)
        torch.save({
            'epoch': epoch,
            'lr': lr,
            'optimizer': optimizer.state_dict(),
            'model_pos': model_pos_train.state_dict(),
        }, chk_path)

        best_result_p1 = errors_3d_valid_p1[-1]
        best_result_p2 = errors_3d_valid_p2[-1]
        best_result_errors_3d_valid_p1_action = copy.deepcopy(errors_3d_valid_p1_action)
        best_result_errors_3d_valid_p2_action = copy.deepcopy(errors_3d_valid_p2_action)

print('-' * 50)
print('Best results per action:')
for action in test_actions:
    print('{}: MPJPE: {:.2f} P-MPJPE: {:.2f}'.format(action.ljust(12, ' '),
                                                     best_result_errors_3d_valid_p1_action[action] * 1000,
                                                     best_result_errors_3d_valid_p2_action[action] * 1000))
print('Best overall MPJPE: {:.2f} P-MPJPE: {:.2f}'.format(best_result_p1, best_result_p2))
