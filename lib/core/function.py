import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

from core.loss import loss_function_ab, loss_function_af
from core.ab_match import anchor_box_adjust, anchor_bboxes_encode
from core.distribution_similarity_via_optimal_transport import SinkhornSolver
from core.utils_ab import result_process_ab, result_process_af
from Evaluation.ActivityNet.eval_detection import ANETdetection
from Evaluation.ActivityNet.eval_classification import ANETclassification

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def ab_prediction_train(cfg, out_ab, label, boxes, action_num):
    '''
    Loss for anchor-based module includes: category classification loss, overlap loss and regression loss
    '''
    match_xs_ls = list()
    match_ws_ls = list()
    match_labels_ls = list()
    match_scores_ls = list()
    anchors_class_ls = list()
    anchors_overlap_ls = list()
    anchors_x_ls = list()
    anchors_w_ls = list()

    for i, layer_name in enumerate(cfg.MODEL.LAYERS_NAME):
        match_xs, match_ws, match_scores, match_labels, \
        anchors_x, anchors_w, anchors_overlap, anchors_class = \
            anchor_bboxes_encode(cfg, out_ab[i], label, boxes, action_num, layer_name)

        match_xs_ls.append(match_xs)
        match_ws_ls.append(match_ws)
        match_scores_ls.append(match_scores)
        match_labels_ls.append(match_labels)

        anchors_x_ls.append(anchors_x)
        anchors_w_ls.append(anchors_w)
        anchors_overlap_ls.append(anchors_overlap)
        anchors_class_ls.append(anchors_class)

    # collect the predictions
    match_xs_ls = torch.cat(match_xs_ls, dim=1)
    match_ws_ls = torch.cat(match_ws_ls, dim=1)
    match_labels_ls = torch.cat(match_labels_ls, dim=1)
    match_scores_ls = torch.cat(match_scores_ls, dim=1)
    anchors_class_ls = torch.cat(anchors_class_ls, dim=1)
    anchors_overlap_ls = torch.cat(anchors_overlap_ls, dim=1)
    anchors_x_ls = torch.cat(anchors_x_ls, dim=1)
    anchors_w_ls = torch.cat(anchors_w_ls, dim=1)

    return anchors_x_ls, anchors_w_ls, anchors_overlap_ls, anchors_class_ls, \
           match_xs_ls, match_ws_ls, match_scores_ls, match_labels_ls


def ab_predict_eval(cfg, out_network, video_name, begin_frame):
    # collect predictions
    anchors_class_ls = list()
    anchors_overlap_ls = list()
    anchors_x_ls = list()
    anchors_w_ls = list()

    for i, layer_name in enumerate(cfg.MODEL.LAYERS_NAME):
        anchors_class, anchors_overlap, anchors_x, anchors_w = anchor_box_adjust(cfg, out_network[i], layer_name)
        anchors_class_ls.append(anchors_class)
        anchors_overlap_ls.append(anchors_overlap)
        anchors_x_ls.append(anchors_x)
        anchors_w_ls.append(anchors_w)

    # classification score
    anchors_class_ls = torch.cat(anchors_class_ls, dim=1)
    # overlap
    anchors_overlap_ls = torch.cat(anchors_overlap_ls, dim=1)
    # regression
    anchors_x_ls = torch.cat(anchors_x_ls, dim=1)
    anchors_w_ls = torch.cat(anchors_w_ls, dim=1)

    # classification score
    m = nn.Softmax(dim=2).cuda()
    anchors_class_ls = m(anchors_class_ls)
    cls_score = anchors_class_ls.detach().cpu().numpy()
    # overlap
    overlap = anchors_overlap_ls.detach().cpu().numpy()
    # regression
    anchors_xmins = anchors_x_ls - anchors_w_ls / 2
    tmp_xmins = anchors_xmins.detach().cpu().numpy()
    xmins = tmp_xmins * cfg.DATASET.WINDOW_SIZE

    anchors_xmaxs = anchors_x_ls + anchors_w_ls / 2
    tmp_xmaxs = anchors_xmaxs.detach().cpu().numpy()
    xmaxs = tmp_xmaxs * cfg.DATASET.WINDOW_SIZE

    video_len = cfg.DATASET.WINDOW_SIZE

    df_prediction = result_process_ab(video_name, video_len, begin_frame, cls_score, overlap, xmins, xmaxs, cfg)

    return df_prediction


def train(cfg, train_loader, model, optimizer):
    model.train()
    loss_record = 0
    cls_loss_record = 0
    overlap_loss_record = 0
    loc_loss_record = 0

    for feat_spa, feat_tem, boxes, label, action_num in train_loader:
        optimizer.zero_grad()

        feat_spa = feat_spa.type_as(dtype)
        feat_tem = feat_tem.type_as(dtype)
        boxes = boxes.float().type_as(dtype)
        label = label.type_as(dtypel)

        pred_cat, cost_ot = model(feat_spa, feat_tem, is_train=True)

        # measure distribution similarity
        loss_ot_distance = cost_ot * cfg.TRAIN.P_OT_WEIGHT

        # concatnate spatial feature and temporal feature
        anchors_x_ls_cat, anchors_w_ls_cat, anchors_overlap_ls_cat, anchors_class_ls_cat, \
        match_xs_ls_cat, match_ws_ls_cat, match_scores_ls_cat, match_labels_ls_cat = ab_prediction_train(cfg, pred_cat, label, boxes, action_num)
        cls_loss_cat, overlap_loss_cat, loc_loss_cat = loss_function_ab(anchors_x_ls_cat, anchors_w_ls_cat, anchors_overlap_ls_cat,
                                                                     anchors_class_ls_cat, match_xs_ls_cat, match_ws_ls_cat, match_scores_ls_cat, match_labels_ls_cat, cfg)
        loss_cat = cls_loss_cat + cfg.TRAIN.P_CONF_AB * overlap_loss_cat + cfg.TRAIN.P_LOC_AB * loc_loss_cat

        loss = loss_cat + loss_ot_distance

        loss.backward()
        optimizer.step()
        loss_record = loss_record + loss.item()

        cls_loss = cls_loss_cat
        cls_loss_record = cls_loss_record + cls_loss.item()
        overlap_loss = overlap_loss_cat
        overlap_loss_record = overlap_loss_record + overlap_loss.item()
        loc_loss = loc_loss_cat
        loc_loss_record = loc_loss_record + loc_loss.item()

    loss_avg = loss_record / len(train_loader)

    cls_loss_record_avg = cls_loss_record / len(train_loader)
    overlap_loss_record_avg = overlap_loss_record / len(train_loader)
    loc_loss_record_avg = loc_loss_record / len(train_loader)

    return loss_avg, cls_loss_record_avg, overlap_loss_record_avg, loc_loss_record_avg


def evaluation(val_loader, model, epoch, cfg):
    model.eval()

    out_df = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS_AB)
    for feat_spa, feat_tem, begin_frame, video_name in val_loader:
        begin_frame = begin_frame.detach().numpy()

        feat_spa = feat_spa.type_as(dtype)
        feat_tem = feat_tem.type_as(dtype)
        out_cat = model(feat_spa, feat_tem, is_train=False)

        pred_cat = ab_predict_eval(cfg, out_cat, video_name, begin_frame)

        out_df = pd.concat([out_df, pred_cat])

    if cfg.BASIC.SAVE_PREDICT_RESULT:
        predict_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.PREDICT_CSV_FILE+'_ab'+str(epoch)+'.csv')
        print('predict_file', predict_file)
        out_df.to_csv(predict_file, index=False)

    return out_df


def evaluate_mAP(cfg, json_path, gt_path, verbose):
    tIoU_thresh = np.array(cfg.TEST.IOU_TH)
    anet_detection = ANETdetection(gt_path, json_path,
                                   subset=cfg.DATASET.VAL_SPLIT, tiou_thresholds=tIoU_thresh,
                                   verbose=verbose, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()

    if verbose:
        for i in range(tIoU_thresh.shape[0]):
            # print(tIoU_thresh[i], mAP[i])
            print(mAP[i])
    return mAP, average_mAP


def evaluate_cls(cfg, prediction_filename, ground_truth_filename, verbose=True):
    anet_classification = ANETclassification(ground_truth_filename,
                                             prediction_filename,
                                             subset=cfg.DATASET.VAL_SPLIT, verbose=verbose,
                                             check_status=False)
    ap, map, top_k, hit_at_k, avg_hit_at_k = anet_classification.evaluate()
    return ap, map, top_k, hit_at_k, avg_hit_at_k
