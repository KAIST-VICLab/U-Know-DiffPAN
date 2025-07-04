import torch.utils.data as data
import torch
import cv2
import os
from PIL import Image
import matplotlib.cm as cm
# from utils.postprocess import convert_to_grayscale, contrast_stretch, to_rgb
import numpy as np
import einops


root = '/hdd/sungpyo/Pan-Sharpening/Pancollection'
wv3_train = 'training_data/train_wv3.h5'
qb_train = 'training_data/train_qb.h5'
gf2_train = 'training_data/train_gf2.h5'

wv3_fr_test = 'test_data/test_wv3_OrigScale_multiExm1.h5'
qb_fr_test = 'test_data/test_qb_OrigScale_multiExm1.h5'
gf2_fr_test = 'test_data/test_gf2_OrigScale_multiExm1.h5'
wv2_fr_test = 'test_data/test_wv2_OrigScale_multiExm1.h5'

wv3_rr_test = 'test_data/test_wv3_multiExm1.h5'
qb_rr_test = 'test_data/test_qb_multiExm1.h5'
gf2_rr_test = 'test_data/test_gf2_multiExm1.h5'
wv2_rr_test = 'test_data/test_wv2_multiExm1.h5'

def rgb_save(output, img_scale, save_dir, file_name):
    x = to_rgb(output)
    x =  torch.Tensor(x.copy())   #BGR
    x = x.clamp_(0,1)
    x = x.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    im  = Image.fromarray(x)
    save_dir = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    im.save(save_path)
    return x

def var_rgb_save(output, save_dir, file_name):
    output = contrast_stretch(output, first_channel=True)
    im  = Image.fromarray(output)
    save_dir = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    im.save(save_path)
    return output



# def save_error_map(error_map, img_scale, save_dir, file_name='error_map.png', colormap_name='virdis'):
def save_error_map(error_map, img_scale, save_dir, file_name='error_map.png', colormap_name='jet'):
    """
    error_map: 계산된 에러 맵 데이터 (PyTorch tensor)
    filename: 저장할 파일 이름
    colormap: 사용할 colormap의 이름
    
    에러 맵을 이미지 파일로 저장합니다.
    """
    # C H W
    error_map = convert_to_grayscale(error_map)
    # H W
    error_map =  torch.Tensor(error_map.copy()).squeeze()   #BGR
    
    # 에러 맵을 CPU로 이동시키고 NumPy 배열로 변환
    error_map_np = error_map.cpu().numpy()

    # Colormap 적용
    normed_data = error_map_np
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

def draw_rectangle(image, top_left, bottom_right, color=(255, 0, 0), thickness=2):
    """Draw a rectangle on the image"""
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

def crop_and_resize(image, top_left, bottom_right, scale=2):
    """Crop and resize the region of interest"""
    cropped = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    resized = cv2.resize(cropped, (0, 0), fx=scale, fy=scale)
    return resized

def place_zoomed_image_on_highlighted(highlighted, zoomed, position='bottom_right'):
    """Place the zoomed image on the bottom right corner of the highlighted image"""
    h_highlighted, w_highlighted, _ = highlighted.shape
    h_zoomed, w_zoomed, _ = zoomed.shape

    if position == 'bottom_right':
        x_offset = w_highlighted - w_zoomed
        y_offset = h_highlighted - h_zoomed
    elif position == 'bottom_left':
        x_offset = 0
        y_offset = h_highlighted - h_zoomed
    elif position == 'top_left':
        x_offset = 0
        y_offset = 0
    else:
        raise ValueError("Unsupported position: Only 'bottom_right' is currently supported")

    result = highlighted.copy()
    result[y_offset:y_offset + h_zoomed, x_offset:x_offset + w_zoomed] = zoomed
    return result

def comparison_image_save(image, top_left, bottom_right, save_dir, file_name, zoom_scale=2, zoom_border_color=(255, 0, 0), zoom_border_thickness=2, position = 'top_left'): 
    # Load the image
    # image = cv2.imread(image_path)
    # H, W, C numpy

    # Copy the image for highlighting
    image = image[:,:,:3]
    highlighted_image = image.copy()
    draw_rectangle(highlighted_image, top_left, bottom_right, color=zoom_border_color, thickness=zoom_border_thickness//2)

    # Crop and resize the region of interest
    zoomed_image = crop_and_resize(image, top_left, bottom_right, scale=zoom_scale)

    # Draw border around zoomed image
    draw_rectangle(zoomed_image, (0, 0), (zoomed_image.shape[1]-zoom_border_thickness//2, zoomed_image.shape[0]-zoom_border_thickness//2), color=zoom_border_color, thickness=zoom_border_thickness)

    # Place the zoomed image on the highlighted image
    # result_image = place_zoomed_image_on_highlighted(highlighted_image, zoomed_image, position= 'bottom_left')
    result_image = place_zoomed_image_on_highlighted(highlighted_image, zoomed_image, position = position)

    img = Image.fromarray(result_image)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    img.save(save_path)
    return result_image

def to_rgb(x, tol_low=0.01, tol_high=0.99):
    x = (x + 1.0) / 2.0
    x = torch.Tensor(x)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() == 3:
        has_batch = False
        x = x.unsqueeze(0)
    else:
        has_batch = True
    # Try to detect BCHW or BHWC
    if x.shape[1] > 8:
        x = einops.rearrange(x, 'b h w c -> b c h w')
    c = x.shape[1]
    if c == 1:
        x = torch.cat([x, x, x], dim=1)
    elif c == 3:
        pass
    elif c == 4:
        x = x[:, [2, 1, 0], :, :]
    elif c == 8:
        x = x[:, [4, 2, 1], :, :]
    else:
        raise ValueError(f"Unsupported channel number: {c}")
    b, c, h, w = x.shape
    x = einops.rearrange(x, 'b c h w -> c (b h w)')
    sorted_x, _ = torch.sort(x, dim=1)
    t_low = sorted_x[:, int(b * h * w * tol_low)].unsqueeze(1)
    t_high = sorted_x[:, int(b * h * w * tol_high)].unsqueeze(1)
    x = torch.clamp((x - t_low) / (t_high - t_low), 0, 1)
    x = einops.rearrange(x, 'c (b h w) -> b h w c', b=b, c=c, h=h, w=w)
    if not has_batch:
        x = x.squeeze(0)
    return x.cpu().numpy()

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
    grayscale_im = np.sum(np.abs(im_as_arr), axis=-1)*(10/C)
    grayscale_im = np.expand_dims(grayscale_im, axis=-1)
    return grayscale_im


def contrast_stretch(image, first_channel):
    """
    Perform contrast stretching on an image.
    """
    if isinstance(image, torch.Tensor):
        if first_channel:
            image = image.permute(1, 2, 0)

        if image.shape[2] == 8:
            output = image[..., [0, 2, 4]]  # BGR
        elif image.shape[2] == 4 or image.shape[2] == 3:
            output = image[..., [0, 1, 2]]  # BGR
        elif image.shape[2] == 1:
            output = image[..., [0, 0, 0]]  # gray
        output = output.cpu().detach().numpy()

    # Get the minimum and maximum pixel values
    p2, p98 = np.percentile(output, (2, 98))
    
    # Stretch the image
    image_stretched = np.clip((output - p2) * 255.0 / (p98 - p2), 0, 255)
    
    return image_stretched.astype(np.uint8)


if __name__ == '__main__':
    from PIL import Image