import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import _init_paths
from config import cfg
from utils.utils import fix_random_seed
from config import update_config
import pprint
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset.TALDataset import TALDataset
from models.network import Network
from core.function import evaluation, evaluate_mAP
from core.post_process import final_result_process
from utils.utils import backup_codes


def parse_args():
    parser = argparse.ArgumentParser(description='SSAD temporal action localization')
    parser.add_argument('--cfg', type=str, help='experiment config file', default='../experiments/thumos/network.yaml')
    parser.add_argument('--weight_file', default='../checkpoint/model_32.pth')
    parser.add_argument('--epoch', default=32)
    parser.add_argument('--gt_json', default='../lib/dataset/materials_THUMOS14/gt_thumos14_augment.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(args.cfg)
    # create output directory
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    cfg.TRAIN.MODEL_DIR = os.path.join(cfg.TRAIN.MODEL_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX)
    cfg.TRAIN.LOG_FILE = os.path.join(cfg.TRAIN.MODEL_DIR, cfg.TRAIN.LOG_FILE)
    cfg.TEST.PREDICT_CSV_FILE = os.path.join(cfg.TRAIN.MODEL_DIR, cfg.TEST.PREDICT_CSV_FILE)
    cfg.TEST.PREDICT_TXT_FILE = os.path.join(cfg.TRAIN.MODEL_DIR, cfg.TEST.PREDICT_TXT_FILE)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    # copy config file
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, 'code')
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LISTS)
    fix_random_seed(cfg.BASIC.SEED)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # data loader
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)

    model = Network(cfg)
    #evaluate existing model
    epoch = args.epoch
    checkpoint = torch.load(args.weight_file)
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    
    out_df_ab = evaluation(val_loader, model, epoch, cfg)
    actions_json_file = final_result_process(out_df_ab, epoch, cfg, flag=1)
    mAP, average_mAP = evaluate_mAP(cfg, actions_json_file, args.gt_json, verbose=True)


if __name__ == '__main__':
    main()


