import pandas as pd
import os
import json

from core.nms import temporal_nms


def get_video_fps(video_name, cfg):
    # determine FPS
    if video_name in cfg.TEST.VIDEOS_25FPS:
        fps = 25
    elif video_name in cfg.TEST.VIDEOS_24FPS:
        fps = 24
    else:
        fps = 30
    return fps


def record_localizations_json(loc_result, result_file):
    '''
    Prepare the output following the ActivityNet format
    '''
    output_dict = {'version': 'VERSION 1.3', 'results': loc_result, 'external_data': {}}
    outfile = open(result_file, 'w')
    json.dump(output_dict, outfile)
    outfile.close()
    return

def final_result_process(out_df, epoch, cfg, flag):
    '''
    flag:
    0: jointly consider out_df_ab and out_df_af
    1: only consider out_df_ab
    2: only consider out_df_af
    '''

    CATEGORY_IDX = [7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79, 85, 92, 93, 97]
    CATEGORY_NAME = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving',
                    'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
                    'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault',
                    'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
    idx_name_dict = {}
    for i in range(len(CATEGORY_IDX)):
        idx_name_dict[CATEGORY_IDX[i]] = CATEGORY_NAME[i]


    if flag == 0:
        df_ab, df_af = out_df
        df_name = df_ab
    elif flag == 1:
        df_ab = out_df
        df_name = df_ab
    elif flag == 2:
        df_af = out_df
        df_name = df_af
    else:
        raise ValueError('flag should in {0, 1, 2}')

    video_name_list = list(set(df_name.video_name.values[:]))

    predictions = dict()

    for video_name in video_name_list:
        preds = list()

        if flag == 0:
            df_ab, df_af = out_df
            tmpdf_ab = df_ab[df_ab.video_name == video_name]
            tmpdf_af = df_af[df_af.video_name == video_name]
            tmpdf = pd.concat([tmpdf_ab, tmpdf_af], sort=True)
        elif flag == 1:
            tmpdf = df_ab[df_ab.video_name == video_name]
        else:
            tmpdf = df_af[df_af.video_name == video_name]

        # assign cliffDiving instance as diving too
        type_set = list(set(tmpdf.cate_idx.values[:]))
        if cfg.TEST.CATE_IDX_OCC in type_set:
            cliff_diving_df = tmpdf[tmpdf.cate_idx == cfg.TEST.CATE_IDX_OCC]
            diving_df = cliff_diving_df
            diving_df.loc[:, 'cate_idx'] = cfg.TEST.CATE_IDX_REP
            tmpdf = pd.concat(([tmpdf, diving_df]))

        df_nms = temporal_nms(tmpdf, cfg)

        # ensure there are most 200 proposals
        df_vid = df_nms.sort_values(by='score', ascending=False)
        fps = get_video_fps(video_name, cfg)

        for i in range(min(len(df_vid), cfg.TEST.TOP_K_RPOPOSAL)):
            start_time = df_vid.start.values[i] / fps
            end_time = df_vid.end.values[i] / fps
            label = df_vid.label.values[i]
            score = df_vid.score.values[i]
            pred = dict()
            pred['label'] = idx_name_dict[int(label)]  #########
            pred['segment'] = [float(start_time), float(end_time)]
            pred['score'] = float(score)
            preds.append(pred)
        predictions[video_name] = preds

    name ='action_detection'
    output_json_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, str(epoch).zfill(3) + '_' + name + '.json')
    record_localizations_json(predictions, output_json_file)
