import torch
import torch.nn as nn
from torch.nn import functional as F

class EstimatorCV():
    def __init__(self, feature_num, class_num, device):
        super(EstimatorCV, self).__init__()

        self.class_num = class_num
        self.device = device
        self.CoVariance = torch.zeros(class_num, feature_num).to(device)
        self.Ave = torch.zeros(class_num, feature_num).to(device)
        self.Amount = torch.zeros(class_num).to(device)

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        label_mask = (labels == 255).long()
        # TODO
        INDEX255_TO_0 = 0
        # labels = ((1 - label_mask).mul(labels) + label_mask * 19).long()
        labels = ((1 - label_mask).mul(labels) + label_mask * INDEX255_TO_0).long()


        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num, device):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num + 1, device)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):
        label_mask = (labels == 255).long()
        labels = (1 - label_mask).mul(labels).long()

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        # TODO
        # weight_m = list(fc.parameters())[0].squeeze()
        weight_m = list(fc.parameters())[0].squeeze().cuda()

        # TODO
        # NxW_ij = weight_m.expand(N, C, A)
        NxW_ij = weight_m.expand(N, C, A).cuda()

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        # !!!!!!!!!!!!!!!!RuntimeError: expected backend CUDA and dtype Float but got backend CUDA and dtype Long
        label_mask = label_mask.float()

        aug_result = y + 0.5 * sigma2.mul((1 - label_mask).view(N, 1).expand(N, C))

        return aug_result

    def forward(self, features, final_conv, y, target_x, ratio):
        # features = model(x)

        N, A, H, W = features.size()

        target_x = target_x.view(N, 1, target_x.size(1), target_x.size(2)).float()

        target_x = F.interpolate(target_x, size=(H, W), mode='nearest', align_corners=None)

        target_x = target_x.long().squeeze()

        C = self.class_num

        features_NHWxA = features.permute(0, 2, 3, 1).contiguous().view(N * H * W, A)

        target_x_NHW = target_x.contiguous().view(N * H * W)

        y_NHWxC = y.permute(0, 2, 3, 1).contiguous().view(N * H * W, C)

        self.estimator.update_CV(features_NHWxA.detach(), target_x_NHW)

        isda_aug_y_NHWxC = self.isda_aug(final_conv, features_NHWxA, y_NHWxC, target_x_NHW,
                                         self.estimator.CoVariance.detach(), ratio)

        isda_aug_y = isda_aug_y_NHWxC.view(N, H, W, C).permute(0, 3, 1, 2)

        return isda_aug_y


if __name__ == '__main__':
    pass

    # feature_x_unsup_l, model.module.branch1.final_conv_1, x_unsup_l, max_r, ratio

    from network_isda import Network
    from utils.init_func import init_weight
    # model.module.branch1.final_conv_1
    model = Network(21, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
                    pretrained_model="../../DATA/pytorch-weight/resnet50_v1c.pth",
                    norm_layer=nn.BatchNorm2d)
    for name, param in model.named_parameters():
        print(name, "***********", param.shape)

    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    feature_x_unsup_l = torch.randn(2, 256, 128, 128)
    x_unsup_l = torch.randn(2, 21, 128, 128)
    max_r = torch.randint(22, (2, 512, 512))
    max_r[max_r==21] = 255
    print(torch.unique(max_r))
    ratio = 7.5 / 40000
    # isda_augmentor_1 = ISDALoss(256, 21, "cuda")
    device = "cuda:0"
    isda_augmentor_1 = ISDALoss(256, 21, device)
    feature_x_unsup_l = feature_x_unsup_l.to(device)
    x_unsup_l = x_unsup_l.to(device)
    max_r = max_r.to(device)
    # feature_x_unsup_l.cuda()
    # x_unsup_l.cuda()
    # max_r.cuda()
    x_isda_l = isda_augmentor_1(feature_x_unsup_l, model.branch1.final_conv_1, x_unsup_l, max_r, ratio)
    print(x_isda_l.shape)