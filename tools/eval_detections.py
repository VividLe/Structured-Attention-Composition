import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import pprint
import time

import _init_paths
from config import cfg
from config import update_config
from core.function import evaluate_mAP, evaluate_cls
from utils.utils import save_best_record_txt


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/thumos/network.yaml')
    parser.add_argument('-name', default='action_detection')
    args = parser.parse_args()
    return args


def post_process(cfg, actions_json_file, writer, best_mAP, info, epoch, name):
    mAP, average_mAP = evaluate_mAP(cfg, actions_json_file, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)
    #
    # cls_ap, cls_map, cls_top_k, cls_hit_at_k, cls_avg_hit_at_k = evaluate_cls(cfg, actions_json_file, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)

    for i in range(len(cfg.TEST.IOU_TH)):
        writer.add_scalar('z_mAP@{}/{}'.format(cfg.TEST.IOU_TH[i], name), mAP[i], epoch)
    writer.add_scalar('Average mAP/{}'.format(name), average_mAP, epoch)

    if cfg.DATASET.NAME == "THUMOS14":
        # use mAP@0.5 as the metric
        mAP_5 = mAP[4]
        if mAP_5 > best_mAP:
            best_mAP = mAP_5
            info = [epoch, average_mAP, mAP]
    elif cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2" or cfg.DATASET.NAME  == 'HACS':
        if average_mAP > best_mAP:
            best_mAP = average_mAP
            info = [epoch, average_mAP, mAP]

    return writer, best_mAP, info


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # path configuration
    # cfg.BASIC.LOG_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'log')
    # cfg.BASIC.BACKUP_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'codes_backup')
    # cfg.TRAIN.OUTPUT_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'output')
    # cfg.TEST.RESULT_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'results')
    # cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    # cfg.BASIC.TIME = '2021-03-20-15-12'
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    cfg.TRAIN.MODEL_DIR = os.path.join(cfg.TRAIN.MODEL_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX)
    # log
    writer = SummaryWriter(log_dir=os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR))

    best_mAP = -1
    info = list()
    # Notice: we will not evaluate the last data
    eval_interval = cfg.TEST.EVAL_INTERVAL
    for i in range(len(eval_interval)-1):
        # check whether t+1 CAS exists
        actions_json_file_next = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, str(eval_interval[i+1]).zfill(3) + '_' + args.name + '.json')
        while not os.path.exists(actions_json_file_next):
            time.sleep(3)

        actions_json_file_cas = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, str(eval_interval[i]).zfill(3) + '_' + args.name + '.json')
        writer, best_mAP, info = post_process(cfg, actions_json_file_cas, writer, best_mAP, info, eval_interval[i], args.name)
    save_best_record_txt(cfg, info, os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, "best_record_{}.txt".format(args.name)))

    writer.close()


if __name__ == '__main__':
    main()
