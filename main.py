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
import SimpleITK as sitk


def get_data(n):
    standard_path = sorted(glob.glob("./AM002_20220812_3691639/standard/*/*.dcm"))
    swift_path = sorted(glob.glob("./AM002_20220812_3691639/swift/*/*.dcm"))
    swift_recon_low_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_low/*/*.dcm"))
    swift_recon_medium_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_medium/*/*.dcm"))

    standard_img = norm_dcm_array(img_to_array(standard_path[n]))
    swift_img = norm_dcm_array(img_to_array(swift_path[n]))
    swift_recon_low_img = norm_dcm_array(img_to_array(swift_recon_low_path[n]))
    swift_recon_medium_img = norm_dcm_array(img_to_array(swift_recon_medium_path[n]))

    standard_img = sitk.GetImageFromArray(standard_img)
    swift_img = sitk.GetImageFromArray(swift_img)
    swift_recon_low_img = sitk.GetImageFromArray(swift_recon_low_img)
    swift_recon_medium_img = sitk.GetImageFromArray(swift_recon_medium_img)

    return standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img


def main():
    standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img = get_data(0)

    temporary_mask = np.zeros([512, 512]).astype(np.uint8)
    temporary_mask[200:250, 200:250] = 1
    temporary_mask = sitk.GetImageFromArray(temporary_mask)

    radio_shape_2d = RadiomicsShape2D(standard_img, temporary_mask)

    function_list = [
        # 'getMeshSurfaceFeatureValue',
        'getPixelSurfaceFeatureValue',
        'getPerimeterFeatureValue',
        'getPerimeterSurfaceRatioFeatureValue',
        'getSphericityFeatureValue',
        'getSphericalDisproportionFeatureValue',
        'getMaximumDiameterFeatureValue',
        'getMajorAxisLengthFeatureValue',
    ]

    for function_name in function_list:
        function = getattr(radio_shape_2d, function_name)
        result = function()
        print('%s Results: ' % function_name, result)


if __name__ == '__main__':
    main()
