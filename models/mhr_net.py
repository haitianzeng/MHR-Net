import torch
import torch.nn as nn
from common.function import so3_exponential_map
from common.camera import *
from common.function import *
from common.loss import *
from torch_batch_svd import svd


class MHR_Net(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.num_joints_in = hparams['n_keypoints']
        self.in_features = 2
        self.num_joints_out = hparams['n_keypoints']
        self.n_fully_connected = hparams['n_fully_connected']
        self.n_layers = hparams['n_layers']
        self.dict_basis_size = hparams['dict_basis_size']
        weight_init_std = hparams['weight_init_std']
        self.w_shape_mean = hparams['w_shape_mean']
        self.w_shape_final = hparams['w_shape_final']

        self.procrust_samples = hparams['procrust_samples']
        self.w_procrust = hparams['w_procrust']
        self.align_margin = hparams['align_margin']
        self.projection_type = hparams['projection_type']
        assert self.projection_type in ['orthographic', 'perspective']

        self.fe_net = nn.Sequential(
            *self.make_trunk(dim_in=self.num_joints_in * 2,
                             n_fully_connected=self.n_fully_connected,
                             n_layers=self.n_layers))

        self.alpha_layer = conv1x1(self.n_fully_connected,
                                   self.dict_basis_size,
                                   std=weight_init_std)
        self.shape_layer = conv1x1(self.dict_basis_size, 3 * self.num_joints_in,
                                   std=weight_init_std)
        self.rot_layer = conv1x1(self.n_fully_connected, 3,
                                 std=weight_init_std)
        self.trans_layer = conv1x1(self.n_fully_connected, 1,
                                   std=weight_init_std)
        self.cycle_consistent = True
        self.z_augment = False
        self.z_augment_angle = 0.2

        self.psi_net = nn.Sequential(
            *self.make_trunk(dim_in=self.num_joints_in * 3,
                             n_fully_connected=self.n_fully_connected,
                             n_layers=self.n_layers))
        self.alpha_layer_psi = conv1x1(self.n_fully_connected,
                                       self.dict_basis_size,
                                       std=weight_init_std)

        self.n_deform_samples = 3
        self.z_dim = 4

        self.deformation_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.n_fully_connected + self.dict_basis_size + self.z_dim,
                      out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=3 * self.num_joints_in, kernel_size=(1, 1)),
        )

    def make_trunk(self,
                   n_fully_connected=None,
                   dim_in=None,
                   n_layers=None,
                   use_bn=True):

        layer1 = ConvBNLayer(dim_in,
                             n_fully_connected,
                             use_bn=use_bn)
        layers = [layer1]

        for l in range(n_layers):
            layers.append(ResLayer(n_fully_connected,
                                   int(n_fully_connected / 4)))

        return layers

    def forward(self, input_2d, align_to_root=False, use_procrust=False):
        assert input_2d.shape[1] == self.num_joints_in
        assert input_2d.shape[2] == self.in_features

        preds = {}
        ba = input_2d.shape[0]
        dtype = input_2d.type()
        input_2d_norm, root = self.normalize_keypoints(input_2d)
        if self.z_augment:
            R_rand = rand_rot(ba,
                              dtype=dtype,
                              max_rot_angle=float(self.z_augment_angle),
                              axes=(0, 0, 1))
            input_2d_norm = torch.matmul(input_2d_norm, R_rand[:, 0:2, 0:2])
        preds['keypoints_2d'] = input_2d_norm
        preds['kp_mean'] = root

        # get the features from backbone network
        input_flatten = input_2d_norm.view(-1, self.num_joints_in * 2)
        feats = self.fe_net(input_flatten[:, :, None, None])

        # predict the basis shape
        shape_coeff = self.alpha_layer(feats)[:, :, 0, 0]
        shape_mean = self.shape_layer(shape_coeff[:, :, None, None])[:, :, 0, 0]

        # sample random gaussian noise z
        if torch.cuda.is_available():
            z = torch.cuda.FloatTensor(ba * self.n_deform_samples, self.z_dim).normal_()
        else:
            z = torch.FloatTensor(ba * self.n_deform_samples, self.z_dim).normal_()

        # create input for the deformation layer
        feat_deform = torch.cat((feats, shape_coeff[:, :, None, None]), dim=1).repeat(self.n_deform_samples, 1, 1, 1)
        feat_deform = torch.cat((feat_deform, z[:, :, None, None]), dim=1)
        deformation = self.deformation_layer(feat_deform)

        # calculate the final shape by adding multiple deforms to the basis shape
        shape_invariant = shape_mean.repeat(self.n_deform_samples, 1) + deformation[:, :, 0, 0]

        shape_mean = shape_mean.reshape(ba, self.num_joints_out, 3)
        shape_invariant = shape_invariant.reshape(ba * self.n_deform_samples, self.num_joints_out, 3)

        # predict the rotation
        R_log = self.rot_layer(feats)[:, :, 0, 0]
        R = so3_exponential_map(R_log)
        T = R_log.new_zeros(ba, 3)  # no global depth offset

        # copy the rotations for all hypotheses and do re-projection
        R_invariant = R.repeat(self.n_deform_samples, 1, 1)
        T_invariant = T.repeat(self.n_deform_samples, 1)
        scale = R_log.new_ones(ba)
        scale_invariant = R_log.new_ones(ba * self.n_deform_samples)
        shape_camera_coord = self.rotate_and_translate(shape_invariant, R_invariant, T_invariant, scale_invariant)

        if self.projection_type == 'orthographic':
            shape_image_coord = shape_camera_coord[:, :, 0:2]
        elif self.projection_type == 'perspective':
            shape_image_coord = shape_camera_coord[:, :, 0:2] / torch.clamp(5 + shape_camera_coord[:, :, 2:3], min=1)

        # do re-projection on the basis shape for intermedia loss
        shape_mean_camera_coord = self.rotate_and_translate(shape_mean, R, T, scale)
        if self.projection_type == 'orthographic':
            shape_mean_image_coord = shape_mean_camera_coord[:, :, 0:2]
        elif self.projection_type == 'perspective':
            shape_mean_image_coord = shape_mean_camera_coord[:, :, 0:2] / torch.clamp(
                5 + shape_mean_camera_coord[:, :, 2:3], min=1)  # Perspective projection

        preds['camera'] = input_2d_norm
        preds['shape_coeff'] = shape_coeff
        preds['shape_mean'] = shape_mean

        # best hypothesis selection by finding the hypothesis with the smallest re-projection error (mpjpe)
        shape_inv_mpjpe = input_2d_norm.unsqueeze(0).repeat(self.n_deform_samples, 1, 1, 1) - \
                          shape_image_coord.reshape(self.n_deform_samples, ba, shape_image_coord.shape[1], shape_image_coord.shape[2])
        shape_inv_mpjpe = torch.norm(shape_inv_mpjpe, dim=(2, 3))
        arg_min = torch.argmin(shape_inv_mpjpe, dim=0)

        shape_image_coord = shape_image_coord.reshape(self.n_deform_samples, ba, shape_image_coord.shape[1], shape_image_coord.shape[2])
        shape_image_coord = shape_image_coord[arg_min, torch.arange(ba), :, :]

        shape_camera_coord = shape_camera_coord.reshape(self.n_deform_samples, ba, shape_camera_coord.shape[1], shape_camera_coord.shape[2])
        shape_camera_coord = shape_camera_coord[arg_min, torch.arange(ba), :, :]

        shape_invariant = shape_invariant.reshape(self.n_deform_samples, ba, shape_invariant.shape[1], shape_invariant.shape[2])
        shape_invariant = shape_invariant[arg_min, torch.arange(ba), :, :]

        preds['shape_invariant'] = shape_invariant
        preds['shape_camera_coord'] = shape_camera_coord

        # calculate the data term loss, including re-projections on the best hypothesis shape and the basis shape
        preds['l_reprojection'] = self.w_shape_final * mpjpe(shape_image_coord, input_2d_norm) + \
                                  self.w_shape_mean * mpjpe(shape_mean_image_coord, input_2d_norm)

        preds['align'] = align_to_root

        if self.cycle_consistent:
            preds['l_cycle_consistent'] = self.cycle_consistent_loss(preds)

        if use_procrust:
            preds['l_procrust'] = self.w_procrust * self.procrustean_loss(shape_invariant)

        return preds

    def cycle_consistent_loss(self, preds):
        shape_mean = preds['shape_mean']
        if preds['align']:
            shape_mean_root = shape_mean - shape_mean[:, 0:1, :]
        else:
            shape_mean_root = shape_mean
        dtype = shape_mean.type()
        ba = shape_mean.shape[0]

        n_sample = 4
        # rotate the canonical point
        # generate random rotation around all axes
        R_rand = rand_rot(ba * n_sample,
                          dtype=dtype,
                          max_rot_angle=3.1415926,
                          axes=(1, 1, 1))

        unrotated = shape_mean_root.view(-1, self.num_joints_out, 3).repeat(n_sample, 1, 1)
        rotated = torch.bmm(unrotated, R_rand)

        rotated = rotated.reshape(ba * n_sample, self.num_joints_out * 3, 1, 1)

        psi_feats = self.psi_net(rotated)
        psi_alpha = self.alpha_layer_psi(psi_feats)
        repred_result = self.shape_layer(psi_alpha)

        repred_result = repred_result.view(ba * n_sample, self.num_joints_out, 3)

        a, b = repred_result, unrotated

        l_cycle_consistent = mpjpe(a, b)

        return l_cycle_consistent

    def procrustean_loss(self, shapes):
        ba, _, n_points = shapes.shape

        if self.procrust_samples * 2 > ba:
            return 0

        randperm = torch.randperm(ba)
        indices_i = randperm[:self.procrust_samples]
        indices_j = randperm[self.procrust_samples:self.procrust_samples * 2]

        shapes_i = shapes[indices_i]
        shapes_j = shapes[indices_j]

        # Aligning shape_j to shape_i using orthogonal Procrustes
        with torch.no_grad():
            svd_mat = torch.bmm(shapes_j.transpose(2, 1), shapes_i)
            # U, s, V = torch.svd(svd_mat)  # can use torch.svd alternatively
            U, s, V = svd(svd_mat)
            rot = torch.bmm(V, U.transpose(2, 1))

        aligned_shapes_j = torch.bmm(rot, shapes_j.transpose(2, 1)).transpose(2, 1)
        ori_diff = torch.norm(shapes_i - shapes_j, p='fro', dim=(1, 2))
        aligned_diff = torch.norm(shapes_i - aligned_shapes_j, p='fro', dim=(1, 2))

        # select positions where difference of aligned shapes are small
        pos = (aligned_diff / torch.norm(shapes_j, p='fro', dim=(1, 2))) < self.align_margin

        if torch.any(pos):
            l_procrust = torch.mean((ori_diff - aligned_diff)[pos])
        else:
            l_procrust = 0

        return l_procrust

    def rotate_and_translate(self, S, R, T, s):
        out = torch.bmm(S, R) + T[:, None, :]
        return out

    def normalize_keypoints(self,
                            kp_loc,
                            rescale=1.):
        # center around the root joint
        kp_mean = kp_loc[:, 0, :]
        kp_loc_norm = kp_loc - kp_mean[:, None, :]
        kp_loc_norm = kp_loc_norm * rescale

        return kp_loc_norm, kp_mean

    def normalize_3d(self, kp):
        ls = torch.norm(kp[:, 1:, :], dim=2)
        scale = torch.mean(ls, dim=1)
        kp = kp / scale[:, None, None] * 0.5
        return kp


def pytorch_ge12():
    v = torch.__version__
    v = float('.'.join(v.split('.')[0:2]))
    return v >= 1.2


def conv1x1(in_planes, out_planes, std=0.01):
    """1x1 convolution"""
    cnv = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=1)

    cnv.weight.data.normal_(0., std)
    if cnv.bias is not None:
        cnv.bias.data.fill_(0.)

    return cnv


class ConvBNLayer(nn.Module):
    def __init__(self, inplanes, planes, use_bn=True, stride=1, ):
        super(ConvBNLayer, self).__init__()

        # do a reasonable init
        self.conv1 = conv1x1(inplanes, planes)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            if pytorch_ge12():
                self.bn1.weight.data.uniform_(0., 1.)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        return out


class ResLayer(nn.Module):
    def __init__(self, inplanes, planes, expansion=4):
        super(ResLayer, self).__init__()
        self.expansion = expansion

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if pytorch_ge12():
            self.bn1.weight.data.uniform_(0., 1.)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if pytorch_ge12():
            self.bn2.weight.data.uniform_(0., 1.)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if pytorch_ge12():
            self.bn3.weight.data.uniform_(0., 1.)
        self.relu = nn.ReLU(inplace=True)
        self.skip = inplanes == (planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.skip:
            out += residual
        out = self.relu(out)

        return out


if __name__ == '__main__':
    from models.utils import parse_param
    hparams = parse_param('projection_type=perspective,w_shape_final=0.2,w_shape_mean=0.8,n_keypoints=17,weight_init_std=0.01,n_fully_connected=1024,n_layers=6,dict_basis_size=10,w_procrust=0.1,procrust_samples=32,align_margin=0.08')

    model = MHR_Net(hparams)
    dummy_input = torch.rand(128, 17, 2)
    output = model(dummy_input, use_procrust=True)
