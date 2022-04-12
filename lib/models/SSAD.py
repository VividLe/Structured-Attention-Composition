import torch.nn as nn
import torch


class SSAD(nn.Module):
    '''
    Action multi-class classification and regression
    input:
    output: class score + conf + location (center & width)
    '''
    def __init__(self, in_channels, cfg):
        super(SSAD, self).__init__()
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.num_pred_value = cfg.DATASET.NUM_CLASSES + 3
        self.lrelu = nn.LeakyReLU()

        # base layer, decrease temporal length
        self.base_conv1 = nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=9, stride=1, padding=4, bias=True)
        self.base_max_pooling1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.base_conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=9, stride=1, padding=4, bias=True)
        self.base_max_pooling2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # main network, extract pyramid features
        self.main_conv1 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True)
        self.main_conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True)
        self.main_conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True)

        # three prediction layers
        num_box = cfg.MODEL.NUM_DBOX['AL1']
        self.pred1 = nn.Conv1d(in_channels=512, out_channels=num_box * self.num_pred_value, kernel_size=3, stride=1, padding=1, bias=True)
        num_box = cfg.MODEL.NUM_DBOX['AL2']
        self.pred2 = nn.Conv1d(in_channels=512, out_channels=num_box * self.num_pred_value, kernel_size=3, stride=1, padding=1, bias=True)
        num_box = cfg.MODEL.NUM_DBOX['AL3']
        self.pred3 = nn.Conv1d(in_channels=512, out_channels=num_box * self.num_pred_value, kernel_size=3, stride=1, padding=1, bias=True)

    def tensor_view(self, data):
        '''
        view the tensor for [batch, 120, depth] to [batch, (depth*5), 24]
        make the prediction (24 values) for each anchor box at the last dimension
        '''
        batch, cha, dep = data.size()
        data = data.view(batch, -1, self.num_pred_value, dep)
        data = data.permute(0, 3, 1, 2).contiguous()
        data = data.view(batch, -1, self.num_pred_value)
        return data

    def forward(self, x):

        # base layer, decrease temporal length
        base_feature1 = self.lrelu(self.base_conv1(x))
        base_feature1 = self.base_max_pooling1(base_feature1)
        base_feature2 = self.lrelu(self.base_conv2(base_feature1))
        base_feature2 = self.base_max_pooling2(base_feature2)

        # main network, extract pyramid features
        feature1 = self.lrelu(self.main_conv1(base_feature2))
        feature2 = self.lrelu(self.main_conv2(feature1))
        feature3 = self.lrelu(self.main_conv3(feature2))

        # three prediction layers
        pred_layer1 = self.pred1(feature1)
        pred_layer1 = self.tensor_view(pred_layer1)
        pred_layer2 = self.pred2(feature2)
        pred_layer2 = self.tensor_view(pred_layer2)
        pred_layer3 = self.pred3(feature3)
        pred_layer3 = self.tensor_view(pred_layer3)

        return pred_layer1, pred_layer2, pred_layer3


if __name__ == '__main__':
    import sys
    sys.path.append('/data1/user6/NeurIPS/1-complete-model/lib')
    from config import cfg, update_config
    cfg_file = '/data1/user6/NeurIPS/1-complete-model/experiments/thumos/SSAD.yaml'
    update_config(cfg_file)

    model = SSAD(in_channels=2048, cfg=cfg)
    data = torch.randn((2, 2048, 128))
    out = model(data)
    for i in out:
        print(i.size())

