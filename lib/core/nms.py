import numpy as np
import pandas as pd

from core.utils_ab import tiou


def temporal_nms(df, cfg):
    '''
    temporal nms
    I should understand this process
    '''

    type_set = list(set(df.cate_idx.values[:]))
    # type_set.sort()

    # returned values
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

    # attention: for THUMOS, a sliding window may contain actions from different class
    for t in type_set:
        label = t
        tmp_df = df[df.cate_idx == t]

        start_time = np.array(tmp_df.xmin.values[:])
        end_time = np.array(tmp_df.xmax.values[:])
        scores = np.array(tmp_df.conf.values[:])

        duration = end_time - start_time
        order = scores.argsort()[::-1]

        keep = list()
        while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):
            i = order[0]
            keep.append(i)
            tt1 = np.maximum(start_time[i], start_time[order[1:]])
            tt2 = np.minimum(end_time[i], end_time[order[1:]])
            intersection = tt2 - tt1
            union = (duration[i] + duration[order[1:]] - intersection).astype(float)
            iou = intersection / union

            inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
            order = order[inds + 1]

        # record the result
        for idx in keep:
            rlabel.append(label)
            rstart.append(float(start_time[idx]))
            rend.append(float(end_time[idx]))
            rscore.append(scores[idx])

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    new_df['label'] = rlabel
    return new_df
