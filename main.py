'''
File: main.py
Project: radiomics-mri
File Created: 2023-04-03 19:56:55
Author: sangminlee
-----
This script ...
Reference
...
'''
import glob
from PIL import Image
from utils import norm_dcm_array, img_to_array
import radiomics
import numpy as np
from radiomics.firstorder import RadiomicsFirstOrder
from radiomics.shape2D import RadiomicsShape2D
from radiomics.gldm import RadiomicsGLDM
from radiomics.glcm import RadiomicsGLCM
import SimpleITK as sitk
import cv2
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering


def slic_mri(np_img: np.ndarray, num_of_segment: int, compactness: float):
    rgb_img = np.tile(np_img[:, :, np.newaxis], [1, 1, 3])
    bilateralled = cv2.bilateralFilter(np.array(rgb_img), -1, 20, 20)
    # equalized = cv2.equalizeHist(bilateralled[:, :, 0])
    # filtered = cv2.pyrMeanShiftFiltering(bilateralled, 5, 2)
    segments = slic(bilateralled, num_of_segment, convert2lab=True, compactness=compactness)
    # f = mark_boundaries(rgb_img, segments, mode='subpixel')
    # Image.fromarray(np.uint8(f * 255)).save('./%d_%d.png' % (num_of_segment, int(compactness)))
    mean_list = []
    std_list = []
    x_list = []
    y_list = []
    for n in range(segments.min(), segments.max() + 1):
        mean_list.append(np.mean(np_img[np.where(segments == n)[0], np.where(segments == n)[1]]))
        std_list.append(np.std(np_img[np.where(segments == n)[0], np.where(segments == n)[1]]))
        y_list.append(np.mean(np.where(segments == n)[0]))
        x_list.append(np.mean(np.where(segments == n)[1]))
    norm_mean = 2 * (np.array(mean_list) - np.mean(mean_list)) / np.std(mean_list)
    norm_std = 0.5 * (np.array(std_list) - np.mean(std_list)) / np.std(std_list)
    norm_y = 1 * (np.array(y_list) - np.mean(y_list)) / np.std(y_list)
    norm_x = 1 * (np.array(x_list) - np.mean(x_list)) / np.std(x_list)
    # cluster = DBSCAN()
    cluster = SpectralClustering(2)
    pred = cluster.fit_predict(np.concatenate(
        [norm_mean[:, np.newaxis], norm_std[:, np.newaxis], norm_y[:, np.newaxis], norm_x[:, np.newaxis]], axis=1))
    # dbscan.fit_predict()
    return pred, segments


def get_data(n, return_np_array: bool = False):
    standard_path = sorted(glob.glob("./AM002_20220812_3691639/standard/*/*.dcm"))
    swift_path = sorted(glob.glob("./AM002_20220812_3691639/swift/*/*.dcm"))
    swift_recon_low_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_low/*/*.dcm"))
    swift_recon_medium_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_medium/*/*.dcm"))

    standard_img = norm_dcm_array(img_to_array(standard_path[n]))
    swift_img = norm_dcm_array(img_to_array(swift_path[n]))
    swift_recon_low_img = norm_dcm_array(img_to_array(swift_recon_low_path[n]))
    swift_recon_medium_img = norm_dcm_array(img_to_array(swift_recon_medium_path[n]))
    if return_np_array:
        return standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img

    standard_img = sitk.GetImageFromArray(standard_img)
    swift_img = sitk.GetImageFromArray(swift_img)
    swift_recon_low_img = sitk.GetImageFromArray(swift_recon_low_img)
    swift_recon_medium_img = sitk.GetImageFromArray(swift_recon_medium_img)

    return standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img


def test_auto_seg():
    for m in range(50):
        standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img = get_data(m, True)
        pred, segments = slic_mri(standard_img, 64 * 64, 10)
        for i in range(pred.min(), pred.max() + 1):
            mask = np.zeros([512, 512, 3]).astype(np.uint8)
            for n in range(segments.min(), segments.max() + 1):
                if pred[n - 1] != i:
                    continue
                mask[np.where(segments == n)[0], np.where(segments == n)[1]] = [255, 0, 0]
            Image.blend(Image.fromarray(standard_img).convert('RGBA'), Image.fromarray(mask).convert('RGBA'), 0.3).save(
                './mask_%d_%d.png' % (m, i))
    return


def test_grad_cam():
    test_grad_cam()
    standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img = get_data(0)

    temporary_mask = np.zeros([512, 512]).astype(np.uint8)
    temporary_mask[200:250, 200:250] = 1
    temporary_mask = sitk.GetImageFromArray(temporary_mask)

    features = RadiomicsFirstOrder(standard_img, temporary_mask)
    # features = RadiomicsGLCM(standard_img, temporary_mask)
    # features = RadiomicsGLDM(standard_img, temporary_mask)
    # features = RadiomicsShape2D(standard_img, temporary_mask)

    if type(features) in [RadiomicsFirstOrder, RadiomicsGLCM, RadiomicsGLDM]:
        features._initCalculation()

    available_function_list = [attr for attr in dir(features) if
                               (attr.startswith('get') and attr.endswith('Value'))]

    for function_name in available_function_list:
        ''' Explanation about getattr usage '''
        ''' If function_name is getPixelSurfaceFeatureValue '''
        ''' "getattr(radio_shape_2d, function_name)" equals to radio_shape_2d.getPixelSurfaceFeatureValue '''
        ''' So, "result = function()" equals to "radio_shape_2d.getPixelSurfaceFeatureValue()" '''
        ''' The reason why I used this way is to handle the target functions as list. '''
        function = getattr(features, function_name)
        try:
            result = function()
            print('%s Results: ' % function_name, result)
        except Exception as e:
            print("%s call failed" % function_name)
            print(e)


def main():
    # test_grad_cam()
    test_auto_seg()


if __name__ == '__main__':
    main()
