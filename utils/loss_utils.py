from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import numpy as np
import cv2
from math import exp, pi


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window

def cvt2gray(image):
    #일단 mean으로 적용
    gray_img = torch.mean(image, 1)    
    return gray_img                        

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_HF(data):  # get high-frequency
    rs = torch.zeros_like(data)
    if rs.ndim == 4:
        for b in range(data.shape[0]):
            for i in range(data.shape[1]):
                rs[b, i, :, :] = data[b, i, :, :] - cv2.boxFilter(data[b, i, :, :], -1, (5, 5))
    elif len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))
    return rs

def get_LF(data):  # get low-frequency
    rs = np.zeros_like(data)
    if rs.ndim == 4:
        for b in range(data.shape[0]):
            for i in range(data.shape[1]):
                rs[b, i, :, :] = cv2.boxFilter(data[b, i, :, :], -1, (5, 5))
    elif len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = cv2.boxFilter(data, -1, (5, 5))
    return rs

def beta_nll_loss(mean, variance, target, beta = 0.0):
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * variance.detach() ** beta

    return loss.mean()

def charbonnier_loss(x, y, epsilon=1e-6):
    return torch.sqrt((x - y) ** 2 + epsilon ** 2)

def crossrestoration_loss(gt, pan, pred, pred_pan):
    # loss between Teacher and student
    w_hrms = 0.5
    b,c,h,w = gt.shape

    pan_extend = pan.repeat(1,c,1,1)
    
    hrms_loss = torch.abs(gt - pred)  
    pan_loss = torch.abs(pan_extend - pred_pan)

    loss = w_hrms * hrms_loss.mean() + (1 - w_hrms) * pan_loss.mean()
    return loss

# 10-18 kd_v2
def kd_loss(gt, teacher_out, pred, teacher_feat, student_feat, teacher_variance):

    alpha = 1.0

    feat_loss_list = []
    for i in [0,1,2,3,4,5]:
        feat_loss_list.append(charbonnier_loss(teacher_feat[i],student_feat[i]))
    # hard_out_loss = hard_loss 
    # soft_out_loss = soft_loss
    feat_loss = sum([f_loss.mean() for f_loss in feat_loss_list])
    
    b,c,h,w = gt.shape
    # wv3
    if c == 8:
        w_hard = 1.0;  w_soft = 0.1;  w_feat = 0.001

    # qb, gf2
    elif c == 4:
        w_hard = 1.0;  w_soft = 0.1;  w_feat = 0.001
    
    # uncertainty = False
    uncertainty = True
    hard_out_loss = (alpha + teacher_variance)*torch.abs(gt - pred) if uncertainty else torch.abs(gt - pred) 
    soft_out_loss = (alpha - teacher_variance)*torch.abs(teacher_out - pred) if uncertainty else torch.abs(teacher_out - pred)

    loss = w_hard * hard_out_loss.mean() + w_soft * soft_out_loss.mean() + w_feat * feat_loss 
    return loss

class CrossRes_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt, pan, pred, pred_pan):
        return crossrestoration_loss(gt, pan, pred, pred_pan)    

class KD_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt, teacher_out, pred, teacher_feature_map, student_feature_map, teacher_variance):
        return kd_loss(gt, teacher_out, pred, teacher_feature_map, student_feature_map, teacher_variance)    

class Beta_nll_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, mean, variance, beta):
        return beta_nll_loss(mean, variance, target, beta)

class HybridL1L2(torch.nn.Module):
    def __init__(self):
        super(HybridL1L2, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.loss = LossWarpper(l1=self.l1, l2=self.l2)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict

class HybridL1SSIM(torch.nn.Module):
    def __init__(self, channel=31, weighted_r=(1.0, 0.1)):
        super(HybridL1SSIM, self).__init__()
        assert len(weighted_r) == 2
        self._l1 = torch.nn.L1Loss()
        self._ssim = SSIMLoss(channel=channel)
        self.loss = LossWarpper(weighted_r, l1=self._l1, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss

class HybridL1SSIMSAM(torch.nn.Module):
    def __init__(self, channel=31, weighted_r=(1.0, 0.1, 0.1)):
        super(HybridL1SSIMSAM, self).__init__()
        assert len(weighted_r) == 3
        self._l1 = torch.nn.L1Loss()
        self._ssim = SSIMLoss(channel=channel)
        self._sam = SAMLoss()
        self.loss = LossWarpper(weighted_r, l1=self._l1, ssim=self._ssim, sam=self._sam)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss

class HybridCharbonnierSSIM(torch.nn.Module):
    def __init__(self, weighted_r, channel=31) -> None:
        super().__init__()
        self._ssim = SSIMLoss(channel=channel)
        self._charb = CharbonnierLoss(eps=1e-4)
        self.loss = LossWarpper(weighted_r, charbonnier=self._charb, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict
    
class FreqLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
    def forward(self, img1, img2):
        # convert to freq space
        fft_img1 = torch.fft.rfft2(img1, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        fft_img2 = torch.fft.rfft2(img2, dim=(-2, -1), norm="ortho")  # b, c, h, w/2+1
        #amplitude 성분
        fft_img1_amp = fft_img1.abs()
        fft_img2_amp = fft_img2.abs()
        #phase 성분
        fft_img1_phase = self.phase_normalize(fft_img1.angle())
        fft_img2_phase = self.phase_normalize(fft_img2.angle())
        # 각 성분에 대해 loss 계산
        loss_amp = self.l1(fft_img1_amp, fft_img2_amp)
        loss_phase = self.l1(fft_img1_phase, fft_img2_phase)

        return 0.5*loss_amp + 0.5*loss_phase
    # phase 를 [-pi, pi] -> [0,1]로 normalize
    def phase_normalize(self, phase):
        # phi = pi
        norm_phase = (phase + pi)/(2*pi)
        norm_phase = torch.clip(norm_phase,min=0,max=1)
        return norm_phase

class HybridL1Frequency(torch.nn.Module):
    def __init__(self, weighted_r=(1.0, 0.1)) -> None:
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.freq = FreqLoss()
        self.loss = LossWarpper(weighted_r, l1=self.l1, freq=self.freq)
        
    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss

class LossWarpper(torch.nn.Module):
    def __init__(self, weighted_ratio=(1.0, 1.0), **losses):
        super(LossWarpper, self).__init__()
        self.names = []
        assert len(weighted_ratio) == len(losses.keys())
        self.weighted_ratio = weighted_ratio
        for k, v in losses.items():
            self.names.append(k)
            setattr(self, k, v)

    def forward(self, pred, gt):
        loss = 0.0
        d_loss = {}
        for i, n in enumerate(self.names):
            l = getattr(self, n)(pred, gt) * self.weighted_ratio[i]
            loss += l
            d_loss[n] = l
        return loss, d_loss


class SSIMLoss(torch.nn.Module):
    def __init__(
        self, win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
    ):
        super(SSIMLoss, self).__init__()
        self.window_size = win_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(win_size, self.channel, win_sigma)
        self.win_sigma = win_sigma

    def forward(self, img1, img2):
        # print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.win_sigma)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def ssim(img1, img2, win_size=11, data_range=1, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(win_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, win_size, channel, size_average)


def elementwise_charbonnier_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return torch.sqrt((input - target) ** 2 + (eps * eps))


class HybridL1L2(nn.Module):
    def __init__(self, cof=10.0):
        super(HybridL1L2, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.cof = cof

    def forward(self, pred, gt):
        return self.l1(pred, gt) / self.cof + self.l2(pred, gt)


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, img1, img2) -> Tensor:
        return elementwise_charbonnier_loss(img1, img2, eps=self.eps).mean()

class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, img1, img2):
        # 입력 이미지의 차원 검증
        if not img1.size() == img2.size():
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim != 4 or img1.shape[1] <= 1:
            raise ValueError("Input dimension should be BxCxHxW and n_channels should be greater than 1")

        # 이미지를 float64로 변환
        img1_ = img1.to(torch.float64)
        img2_ = img2.to(torch.float64)

        # 내적과 스펙트럼 노름 계산
        inner_product = (img1_ * img2_).sum(dim=1)  # BxHxW
        img1_spectral_norm = torch.sqrt((img1_ ** 2).sum(dim=1))
        img2_spectral_norm = torch.sqrt((img2_ ** 2).sum(dim=1))

        # 코사인 유사도 계산과 손실 계산을 위한 수치 안정성 보장
        eps = torch.finfo(torch.float64).eps
        # eps = 0
        cos_theta = inner_product / (img1_spectral_norm * img2_spectral_norm + eps)
        cos_theta = torch.clamp(cos_theta, min=0, max=1)

        # 1에서 코사인 유사도를 빼서 손실 계산
        loss = 1 - cos_theta

        # 배치 내 모든 이미지의 평균 손실을 반환
        return loss.mean()

class Spectral1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam = SAMLoss()
    # ms 그리고 fused img를 input으로 받음
    def forward(self, ms, fused):
        degraded = torch.nn.functional.interpolate(fused, size=(ms.size(2), ms.size(3)),
                                              mode="bilinear", align_corners=True)
        loss = self.sam(ms, degraded)
        return loss

# class UIQILoss(nn.Moule):
#     def __init__(self):
#         super().__init__()

#     def forward(self, img_base, img_out):
#         #CC 부분 계산
#         _, channel, h, w = img_out.shape
#         C1 = torch.sum(torch.sum(img_base * img_out, 0), 0) - h * w * (
#         torch.mean(torch.mean(img_base, 0), 0) * torch.mean(torch.mean(img_out, 0), 0))
#         C2 = torch.sum(torch.sum(img_out ** 2, 0), 0) - h * w * (torch.mean(torch.mean(img_out, 0), 0) ** 2)
#         C3 = torch.sum(torch.sum(img_base ** 2, 0), 0) - h * w * (torch.mean(torch.mean(img_base, 0), 0) ** 2)
#         CC = C1 / ((C2 * C3) ** 0.5)
#         CC = torch.mean(CC)
        
#         # 중간 파트


#         # 마지막 부분 계산
#         stdx = C2**0.5
#         stdy = C3**0.5
#         third_term = (2*stdx*stdy)/(C2+C3)
#         return third_term

def get_loss(loss_type):
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "hybrid":
        criterion = HybridL1L2()
    elif loss_type == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "l1ssim":
        criterion = HybridL1SSIM(channel=8, weighted_r=(1.0, 0.1))
    elif loss_type == "charbssim":
        criterion = HybridCharbonnierSSIM(channel=31, weighted_r=(1.0, 1.0))
    else:
        raise NotImplementedError(f"loss {loss_type} is not implemented")
    return criterion


if __name__ == "__main__":
    # loss = SSIMLoss(channel=31)
    # loss = CharbonnierLoss(eps=1e-3)
    loss = Spectral1Loss()
    fused = torch.randn(1, 8, 64, 64, requires_grad=True)
    pan = fused[:,0,...].unsqueeze(0)
    # y = x + torch.randn(1, 8, 64, 64) / 10
    l = loss(pan, fused)
    l.backward()
    print(l)
    print(fused.grad)

