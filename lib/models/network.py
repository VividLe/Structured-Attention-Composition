import torch.nn as nn
import torch
from models.SSAD import SSAD
from models.SinkHorn import SinkhornDistance
import time


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.temporal_length = 128
        self.num_channel = 2048
        self.num_sel = int(round(self.temporal_length * cfg.TRAIN.FEAT_TOPK_RATIO))
        self.smooth_ratio = cfg.MODEL.SMOOTH_RATIO
        self.network_cat = SSAD(in_channels=2048, cfg=cfg)
        self.spa_sim_conv1 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.spa_sim_conv2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.tem_sim_conv1 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.tem_sim_conv2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)

        self.frame_prob = nn.Conv1d(in_channels=2048, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.mod_prob = nn.Conv1d(in_channels=2048, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
        self.cost_matrix = nn.Conv1d(in_channels=2048, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)

        self.lrelu = nn.LeakyReLU()
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=50)
        self.sigmoid = nn.Sigmoid()
        # for smooth loss
        self.BCELoss = nn.BCELoss()

    def cal_ot(self, mode_prob, frame_prob, cost_matrix):
        mu = mode_prob[:, :, 0]
        nu = frame_prob[:, 0, :]
        cost, pi = self.sinkhorn(mu, nu, cost_matrix)
        loss = torch.mean(cost)
        return loss

    def topk_sel_feat(self, feat):
        feat_l2_norm = torch.norm(feat, p='fro', dim=1, keepdim=True)  # [B, 1, T]
        idxs = torch.topk(feat_l2_norm, k=self.num_sel, dim=2)[1]  # [B, 1, N]
        idxs_exp = idxs.expand([-1, self.num_channel, -1])
        feat_sel = torch.gather(feat, dim=2, index=idxs_exp)  # [B, D, N]
        # aggregated feature
        feature = torch.mean(feat_sel, dim=2, keepdim=True)  # [B, D]
        return feature

    def cal_smooth_loss(self, cost_matrix, ratio):
        '''
        attention: [B, 2, T]
        '''
        num_k = int(round((cost_matrix.size(2) - 1) * ratio))
        vector = torch.sum(cost_matrix, dim=1)  # [B, T]
        diff_abs = torch.abs(vector[:, :-1] - vector[:, 1:])
        # amplitude
        amp = torch.mean(torch.topk(diff_abs, num_k, largest=False)[0], dim=1)
        # calculate BCELoss, to decrease the differences
        label = torch.zeros_like(amp)
        loss = self.BCELoss(amp, label)
        return loss

    def forward(self, in_feature_spatial, in_feature_temporal, is_train):
        # convert spatial feature into similarity and specific
        feat_spa_sim = self.lrelu(self.spa_sim_conv1(in_feature_spatial))
        feat_spa_sim = self.lrelu(self.spa_sim_conv2(feat_spa_sim))
        feat_tem_sim = self.lrelu(self.tem_sim_conv1(in_feature_temporal))
        feat_tem_sim = self.lrelu(self.tem_sim_conv2(feat_tem_sim))

        feat_cat = torch.cat([feat_spa_sim, feat_tem_sim], dim=1)  # [B, 2D, T]
        frame_prob = self.sigmoid(self.frame_prob(feat_cat))  # [B, 1, T]
        # feat_mod = torch.mean(feat_cat, dim=2, keepdim=True)  # [B, 2D, 1]
        feat_mod = self.topk_sel_feat(feat_cat)
        mode_prob = self.softmax_dim1(self.mod_prob(feat_mod))  # [B, 2, 1]
        cost_matrix = self.sigmoid(self.cost_matrix(feat_cat))

        weights = torch.matmul(mode_prob, frame_prob)  # [B, 2, T]
        spa_sim_weight = weights[:, :1, :]
        tem_sim_weight = weights[:, 1:, :]
        feat_spa_sim = feat_spa_sim * spa_sim_weight
        feat_tem_sim = feat_tem_sim * tem_sim_weight

        # similarity predictions
        feature_sim = torch.cat([feat_spa_sim, feat_tem_sim], dim=1)
        pred_cat = self.network_cat(feature_sim)

        if not is_train:
            return pred_cat
        else:
            ot_loss = self.cal_ot(mode_prob, frame_prob, cost_matrix)
            # constrain sum of the cost matrix should be smooth
            smooth_loss = self.cal_smooth_loss(cost_matrix, self.smooth_ratio)
            loss = ot_loss + smooth_loss
            return pred_cat, loss


if __name__ == '__main__':
    import sys
    sys.path.append('/data1/user6/NeurIPS/1-complete-model/lib')
    from SSAD import SSAD
    from SinkHorn import SinkhornDistance
    from config import cfg, update_config
    cfg_file = '/data1/user6/NeurIPS/1-complete-model/experiments/thumos/SSAD.yaml'
    update_config(cfg_file)
    from utils.utils import fix_random_seed
    import torch.backends.cudnn as cudnn

    fix_random_seed(cfg.BASIC.SEED)
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    model = Network(cfg).cuda()
    feature_spatial = torch.randn((4, 1024, 128)).cuda()
    feature_temporal = torch.randn((4, 1024, 128)).cuda()
    out = model(feature_spatial, feature_temporal)
    for preds in out:
        for pred in preds:
            print(pred.size())
