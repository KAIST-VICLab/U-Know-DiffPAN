import torch
import numpy as np
import os
from scipy import io as scio
from PIL import Image
import matplotlib.cm as cm


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """

    if isinstance(im_as_arr, torch.Tensor):
        im_as_arr = im_as_arr.permute(1,2,0)
        im_as_arr = im_as_arr.cpu().detach().numpy()
    H,W,C = im_as_arr.shape
    # grayscale_im = np.sum(np.abs(im_as_arr), axis=-1)*(12/C)
    grayscale_im = np.sum(np.abs(im_as_arr), axis=-1)*(10/C)
    # im_max = np.percentile(grayscale_im, 99)
    # im_min = np.min(grayscale_im)
    # grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=-1)
    return grayscale_im

def save_error_map(error_map, save_dir, file_name='error_map.png', colormap_name='jet'):
    """
    error_map: 계산된 에러 맵 데이터 (PyTorch tensor)
    filename: 저장할 파일 이름
    colormap: 사용할 colormap의 이름
    
    에러 맵을 이미지 파일로 저장합니다.
    """
    # C H W
    error_map = convert_to_grayscale(error_map).squeeze()

    normed_data = error_map
    # mapped_data = cm.get_cmap(colormap_name)[:, :, :3]  # RGBA에서 RGB만 사용
    color_map = cm.get_cmap(colormap_name)
    mapped_data = color_map(normed_data)  # RGBA에서 RGB만 사용
    # H W C

    # 0-255 스케일링 및 uint8로 변환
    mapped_data = (255 * mapped_data).astype('uint8')

    # PIL 이미지로 변환 및 저장
    img = Image.fromarray(mapped_data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    img.save(save_path)
    return mapped_data


def save_error_map_from_mat(file_path, save_dir):

    data = scio.loadmat(file_path)
    img_set = data['sr']
    img_gt_set = data['gt']

    # mat 파일에 저장된 값의 범위가 0~1 이면,  img_scale = 1로 바꿀것
    if 'wv3' in file_path:
        img_scale = 2047.0
        save_dir = os.path.join(save_dir,'WV3')
    elif 'qb' in file_path:
        img_scale = 2047.0
        save_dir = os.path.join(save_dir,'QB')
    else:
        img_scale = 1023.0
        save_dir = os.path.join(save_dir,'GF2')

    for i, np_img in enumerate(img_set):
        tensor_img = torch.tensor(np_img)
        gt_tensor_img = torch.tensor(img_gt_set[i])
        error_map_tensor = torch.abs(tensor_img-gt_tensor_img) / torch.tensor(img_scale)
        error_map_tensor = save_error_map(error_map_tensor, save_dir=os.path.join(save_dir,'errormap'), file_name = f'testimg{i}.png')
        
if __name__ == '__main__':
    file_path = '/home/ksp/Pansharpening/CANConv_results/mat_results/test_save_wv3_multiExm1.mat'
    save_dir = '/home/ksp/Pansharpening/CANConv_results/save_error_map_from_mat_test'
    save_error_map_from_mat(file_path, save_dir)
   
