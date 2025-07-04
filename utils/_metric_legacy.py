#####
# copy from https://github.com/XiaoXiao-Woo/PanCollection/blob/dev/UDL/pansharpening/common/evaluate.py
# thanks a lot
#####

import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from scipy import ndimage
import decimal


# 由于dat及其方差等数值舍入存在误差，最终结果有0.001左右的误差
def q2n(gt, x, q_blocks_size, q_shift):
    '''
    '''
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
        x = x.cpu().numpy()

    N, N1, N2, N3 = gt.shape  # 255 255 8
    size2 = q_blocks_size  # 32

    stepx = math.ceil(N1 / q_shift)  # 8
    stepy = math.ceil(N2 / q_shift)  # 8

    if stepy <= 0:
        stepy = 1
        stepx = 1

    est1 = (stepx - 1) * q_shift + q_blocks_size - N1  # 1
    est2 = (stepy - 1) * q_shift + q_blocks_size - N2  # 1
    # if np.sum(np.array([est1 != 0, est2 != 0])) > 0:
    # refref = np.zeros(shape=[N1+1, N2+1])
    # fusfus = refref.copy()

    for i in range(N3):
        a1 = gt[..., 0]

        ia1 = np.zeros(shape=[N, N1 + est1, N2 + est2])
        ia1[:, : N1, : N2] = a1
        ia1[:, :, N2:N2 + est2] = ia1[:, :, N2 - 1:-1:N2 - est2 + 1]
        ia1[:, N1:N1 + est1, ...] = ia1[:, N1 - 1:-1:N1 - est1 + 1, ...]
        if i == 0:
            refref = ia1[..., np.newaxis]  # np.concatenate(refref, ia1, axis=3)
        else:
            refref = np.concatenate([refref, ia1[..., np.newaxis]], axis=-1)
        if i < N3:
            gt = gt[..., 1:]

    gt = refref

    for i in range(N3):

        a2 = x[..., 0]
        ia2 = np.zeros(shape=[N, N1 + est1, N2 + est2])
        ia2[:, : N1, : N2] = a2
        ia2[:, :, N2:N2 + est2] = ia2[:, :, N2 - 1:-1:N2 - est2 + 1]
        ia2[:, N1:N1 + est1, ...] = ia2[:, N1 - 1:-1:N1 - est1 + 1, ...]
        if i == 0:
            fusfus = ia2[..., np.newaxis]  # np.concatenate(refref, ia1, axis=3)
        else:
            fusfus = np.concatenate([fusfus, ia2[..., np.newaxis]], axis=-1)

        if i < N3:
            x = x[..., 1:]
    x = fusfus

    x = np.array(x, dtype=np.uint16)
    gt = np.array(gt, dtype=np.uint16)

    _, N1, N2, N3 = gt.shape

    if math.ceil(math.log2(N3)) - math.log2(N3) != 0:
        Ndif = pow(2, math.ceil(math.log2(N3))) - N3
        dif = np.zeros(shape=[N, N1, N2, Ndif], dtype=np.uint16)
        gt = np.concatenate(gt, dif, axis=-1)
        x = np.concatenate(x, dif, axis=-1)

    _, _, _, N3 = gt.shape

    valori = np.zeros(shape=[N, stepx, stepy, N3])

    for j in range(stepx):
        for i in range(stepy):
            o = onions_quality(gt[:, j * q_shift:j * q_shift + q_blocks_size,
                               i * q_shift: i * q_shift + size2, :],
                               x[:, j * q_shift:j * q_shift + q_blocks_size,
                               i * q_shift: i * q_shift + size2, :],
                               q_blocks_size)
            # 0.971379489438014	0.00553590637316723	0.00305237797490489	-0.0188289323262161	-0.00420556598390016	-0.0173947468044076	-0.0202144450367593	0.0102693855205061
            valori[:, j, i, :] = o
    q2n_idx_map = np.sqrt(np.sum(valori ** 2, axis=-1))
    # q2n_index = np.mean(q2n_idx_map)
    return q2n_idx_map


def norm_blocco(x, eps=1e-8):
    a = x.mean()
    c = x.std()
    if c == 0:
        c = eps
    return (x - a) / c + 1, a, c


def onions_quality(dat1, dat2, size1):
    dat1 = np.float64(dat1)
    dat2 = np.float64(dat2)

    dat2 = np.concatenate([dat2[..., 0, np.newaxis], -dat2[..., 1:]], axis=-1)
    N, _, _, N3 = dat1.shape
    size2 = size1

    # Block norm
    '''
            319.6474609375 37.05174450544686
            357.970703125 61.54042371537683
            518.708984375 111.53732768071865
            608.23828125 154.26606056030568
            593.412109375 163.97722215177643
            554.8486328125 113.96758695803403
            690.16015625 151.29524031046248
            442.2314453125 94.12877724960003
            mat
              319.6475   37.0698

              357.9707   61.5705

              518.7090  111.5918

              608.2383  154.3414

              593.4121  164.0573

              554.8486  114.0233

              690.1602  151.3692

              442.2314   94.1748
            '''
    for i in range(N3):
        a1, s, t = norm_blocco(np.squeeze(dat1[..., i]))
        # print(s,t)
        dat1[..., i] = a1
        if s == 0:
            if i == 0:
                dat2[..., i] = dat2[..., i] - s + 1
            else:
                dat2[..., i] = -(-dat2[..., i] - s + 1)
        else:
            if i == 0:
                dat2[..., i] = ((dat2[..., i] - s) / t) + 1
            else:
                dat2[..., i] = -(((-dat2[..., i] - s) / t) + 1)
    m1 = np.zeros(shape=[N, N3])
    m2 = m1.copy()

    mod_q1m = 0
    mod_q2m = 0
    mod_q1 = np.zeros(shape=[size1, size2])
    mod_q2 = np.zeros(shape=[size1, size2])

    for i in range(N3):
        m1[..., i] = np.mean(np.squeeze(dat1[..., i]))
        m2[..., i] = np.mean(np.squeeze(dat2[..., i]))
        mod_q1m += m1[..., i] ** 2
        mod_q2m += m2[..., i] ** 2
        mod_q1 += np.squeeze(dat1[..., i]) ** 2
        mod_q2 += np.squeeze(dat2[..., i]) ** 2

    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)

    termine2 = mod_q1m * mod_q2m  # 7.97
    termine4 = mod_q1m ** 2 + mod_q2m ** 2  #
    int1 = (size1 * size2) / (size1 * size2 - 1) * np.mean(mod_q1 ** 2)
    int2 = (size1 * size2) / (size1 * size2 - 1) * np.mean(mod_q2 ** 2)
    termine3 = int1 + int2 - (size1 * size2) / ((size1 * size2 - 1)) * (mod_q1m ** 2 + mod_q2m ** 2)  # 17.8988  ** 2
    mean_bias = 2 * termine2 / termine4  # 1
    if termine3 == 0:
        q = np.zeros(shape=[N, 1, N3])
        q[:, :, N3 - 1] = mean_bias
    else:
        cbm = 2 / termine3
        # 32 32 8
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1.reshape(-1), m2.reshape(-1))
        qv = np.zeros(shape=[N, N3])
        for i in range(N3):
            qv[..., i] = (size1 * size2) / ((size1 * size2) - 1) * np.mean(np.squeeze(qu[:, :, i]))
        q = qv - (size1 * size2) / ((size1 * size2) - 1) * qm
        q = q * mean_bias * cbm
    return q


def onion_mult2D(onion1, onion2):
    _, _, _, N3 = onion1.shape

    if N3 > 1:
        L = N3 // 2
        a = onion1[..., : L]
        b = onion1[..., L:]
        b = np.concatenate([b[..., 0, np.newaxis], -b[..., 1:]], axis=-1)
        c = onion2[..., : L]
        d = onion2[..., L:]
        d = np.concatenate([d[..., 0, np.newaxis], -d[..., 1:]], axis=-1)

        if N3 == 2:
            ris = np.concatenate([a * c - d * b, a * d + c * b], axis=-1)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, np.concatenate([b[..., 0, np.newaxis], -b[..., 1:]], axis=-1))
            ris3 = onion_mult2D(np.concatenate([a[..., 0, np.newaxis], -a[..., 1:]], axis=-1), d)
            ris4 = onion_mult2D(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate([aux1, aux2], axis=-1)
    else:
        ris = onion1 * onion2
    return ris


def onion_mult(onion1, onion2):
    # _, N = onion1.shape
    N = len(onion1)
    if N > 1:

        L = N // 2
        a = onion1[:L]
        b = onion1[L:]
        # b[1:] = -b[1:]
        b = np.append(np.array(b[0]), -b[1:])
        c = onion2[:L]
        d = onion2[L:]
        # d[1:] = -d[1:]
        d = np.append(np.array(d[0]), -d[1:])

        if N == 2:
            ris = np.append(a * c - d * b, a * d + c * b)
        else:

            ris1 = onion_mult(a, c)
            # b[1:] = -b[1:]
            ris2 = onion_mult(d, np.append(np.array(b[0]), -b[1:]))
            # a[1:] = -a[1:]
            ris3 = onion_mult(np.append(np.array(a[0]), -a[1:]), d)
            ris4 = onion_mult(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.append(aux1, aux2)
    else:
        ris = np.array(onion1).reshape(-1) * np.array(onion2).reshape(-1)
    return ris


def compute_index(img_base, img_out, ratio):
    h = img_out.shape[0]
    w = img_out.shape[1]
    chanel = img_out.shape[2]
    # 计算SAM
    sum1 = torch.sum(img_base * img_out, 2)
    sum2 = torch.sum(img_base * img_base, 2)
    sum3 = torch.sum(img_out * img_out, 2)
    t = (sum2 * sum3) ** 0.5
    numlocal = torch.gt(t, 0)
    num = torch.sum(numlocal)
    t = sum1 / t
    angle = torch.acos(t)
    sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num
    SAM = averangle * 180 / 3.14159256

    # 计算ERGAS
    summ = 0
    for i in range(chanel):
        a1 = torch.mean((img_base[:, :, i] - img_out[:, :, i]) ** 2)
        m1 = torch.mean(img_base[:, :, i])
        a2 = m1 * m1
        summ = summ + a1 / a2
    ERGAS = 100 * (1 / ratio) * ((summ / chanel) ** 0.5)

    return SAM, ERGAS


decimal.getcontext().rounding = "ROUND_HALF_UP"
n_digits = 6


# panHrnet: 2.6565  |1.4651  | 0.98364  | 0.98024  | 0.98089-Q8
def analysis_accu(img_base, img_out, ratio, flag_cut_bounds=True, dim_cut=1, choices=4):
    if flag_cut_bounds:
        img_base = img_base[dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut, :]  #:
        img_out = img_out[dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut, :]  #:

    # q2n
    # q2n_index = q2n(img_base, img_out, q_blocks_size=32, q_shift=32)

    h = img_out.shape[0]
    w = img_out.shape[1]
    chanel = img_out.shape[2]

    # 计算SAM
    sum1 = torch.sum(img_base * img_out, 2)
    sum2 = torch.sum(img_base * img_base, 2)
    sum3 = torch.sum(img_out * img_out, 2)
    t = (sum2 * sum3) ** 0.5
    numlocal = torch.gt(t, 0)
    num = torch.sum(numlocal)
    t = sum1 / t
    angle = torch.acos(t)
    sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num

    # 或者采用https://segmentfault.com/a/1190000018929994修改精度
    # averangle = math.ceil(averangle * 1000000) / 1000000
    averangle = (averangle * 10 ** n_digits).round() / (10 ** n_digits)
    # SAM = decimal.Decimal(averangle.cpu().numpy() * 180 / 3.14159256).quantize(decimal.Decimal("0.00000"))
    SAM = averangle * 180 / 3.14159256

    # 计算ERGAS
    summ = 0
    for i in range(chanel):
        a1 = torch.mean((img_base[:, :, i] - img_out[:, :, i]) ** 2)
        m1 = torch.mean(img_base[:, :, i])
        a2 = m1 * m1
        summ = summ + a1 / a2
    ERGAS = 100 * (1 / ratio) * ((summ / chanel) ** 0.5)

    # 计算PSNR
    mse = torch.mean((img_base - img_out) ** 2, 0)
    mse = torch.mean(mse, 0)
    rmse = mse ** 0.5
    temp = torch.log(1 / rmse) / math.log(10)
    PSNR = 20 * temp

    # 计算SSIM
    # img_base = img_base.permute(2, 0, 1)
    # img_out = img_out.permute(2, 0, 1)
    # img_base = img_base.unsqueeze(0)
    # img_out = img_out.unsqueeze(0)
    # SSIM = _ssim(img_base.permute(2, 0, 1).unsqueeze(0), img_out.permute(2, 0, 1).unsqueeze(0))

    # index = torch.zeros((5, chanel + 1))
    # index[0, 1:chanel + 1] = CC
    # index[1, 1:chanel + 1] = PSNR
    # index[2, 1:chanel + 1] = SSIM
    # index[0, 0] = torch.mean(CC)
    # index[1, 0] = torch.mean(PSNR)
    # index[2, 0] = torch.mean(SSIM)
    # index[3, 0] = SAM
    # index[4, 0] = ERGAS

    PSNR = torch.mean(PSNR)
    # SSIM = torch.mean(SSIM)
    # q2n_index = np.mean(q2n_index)

    if choices == 5:
        # 计算CC
        C1 = torch.sum(torch.sum(img_base * img_out, 0), 0) - h * w * (
                torch.mean(torch.mean(img_base, 0), 0) * torch.mean(torch.mean(img_out, 0), 0))
        C2 = torch.sum(torch.sum(img_out ** 2, 0), 0) - h * w * (torch.mean(torch.mean(img_out, 0), 0) ** 2)
        C3 = torch.sum(torch.sum(img_base ** 2, 0), 0) - h * w * (torch.mean(torch.mean(img_base, 0), 0) ** 2)
        CC = C1 / ((C2 * C3) ** 0.5)
        CC = torch.mean(CC)
        return {'SAM': SAM, 'ERGAS': ERGAS, 'PSNR': PSNR, 'CC': CC}  # , q2n_index

    return {'SAM': SAM, 'ERGAS': ERGAS, 'PSNR': PSNR, }


def _ssim(img1, img2):
    img1 = img1.float()
    img2 = img2.float()

    channel = img1.shape[1]
    max_val = 1
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11
    window = create_window(window_size, sigma, channel).cuda()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
    t = ssim_map.shape
    return ssim_map.mean(2).mean(2)

def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size**2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size/2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
#    print(mu1_mu2.shape)
    #print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    
#    print(np.min(sigma1_sq + sigma2_sq), np.min(mu1_sq + mu2_sq))
    
    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
#    print(np.mean(qindex_map))
    
#    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
#    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
#    # sigma !=0 and mu == 0
#    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
#    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
#    # sigma != 0 and mu != 0
#    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) != 0)
#    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
#        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
    return np.mean(qindex_map)

####################
# observation model
####################


def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std)**2) * np.exp(-0.5 * (t2 / std)**2) 
    return w


def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w


def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h


def GNyq2win(GNyq, scale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    #fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)


def mtf_resize(img, satellite='QuickBird', scale=4):
    # satellite GNyq
    scale = int(scale)
    if satellite == 'QuickBird':
        GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif satellite == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    else:
        raise NotImplementedError('satellite: QuickBird or IKONOS')
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float64)
    if img_.ndim == 2:  # Pan
        H, W = img_.shape
        lowpass = GNyq2win(GNyqPan, scale, N=41)
    elif img_.ndim == 3:  # MS
        H, W, _ = img.shape
        lowpass = [GNyq2win(gnyq, scale, N=41) for gnyq in GNyq]
        lowpass = np.stack(lowpass, axis=-1)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (H // scale, W // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_

##################
# No reference IQA
##################


def D_lambda(img_fake, img_lm, block_size=32, p=1):
    """Spectral distortion
    img_fake, generated HRMS
    img_lm, LRMS"""
    assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i+1, C_f):
            # for fake
            band1 = img_fake[..., i]
            band2 = img_fake[..., j]
            Q_fake.append(_qindex(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[..., i]
            band2 = img_lm[..., j]
            Q_lm.append(_qindex(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()
    return D_lambda_index ** (1/p)


def D_s(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == scale, 'Spatial resolution should be compatible with scale'
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # fake and pan
    assert pan.ndim == 3, 'Panchromatic image must be 3D!'
    H_p, W_p, C_p = pan.shape
    assert C_p == 1, 'size of 3rd dim of Panchromatic image must be 1'
    assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
    # get LRPan, 2D
    pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
    #print(pan_lr.shape)
    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[..., i]
        band2 = pan[..., 0] # the input PAN is 3D with size=1 along 3rd dim
        #print(band1.shape)
        #print(band2.shape)
        Q_hr.append(_qindex(band1, band2, block_size=block_size))
        band1 = img_lm[..., i]
        band2 = pan_lr  # this is 2D
        #print(band1.shape)
        #print(band2.shape)
        Q_lr.append(_qindex(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
    return D_s_index ** (1/q)

def qnr(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    """QNR - No reference IQA"""
    D_lambda_idx = D_lambda(img_fake, img_lm, block_size, p)
    D_s_idx = D_s(img_fake, img_lm, pan, satellite, scale, block_size, q)
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    # print(img_fake)
    return QNR_idx


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def compare_index(A):
    A_size = A.shape
    ite_n = A_size[2]
    band_n = A_size[1]
    C_better = A[:, 0, 0]
    ind = 0
    for i in range(ite_n):
        score_b = 0
        score_c = 0
        C_compare = A[:, 0, i]
        if (C_better[0] > C_compare[0]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1
        if (C_better[1] > C_compare[1]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1
        if (C_better[2] > C_compare[2]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1
        if (C_better[3] < C_compare[3]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1
        if (C_better[4] < C_compare[4]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1

        if (score_c > score_b):
            C_better = A[:, 0, i]
            ind = i

    C_best = A[:, :, ind]
    best_ind = ind + 1
    return C_best, best_ind


if __name__ == "__main__":
    a = torch.randn(256, 256, 3)
    b = torch.randn(256, 256, 3)
    print(analysis_accu(a, b, 1, choices=5))
