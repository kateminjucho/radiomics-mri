'''
File: main.py
Project: radiomics-mri
File Created: 2023-04-03 19:56:55
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
import pandas as pd
import re


def slic_mri(np_img: np.ndarray, num_of_segment: int, compactness: float):
    rgb_img = np.tile(np_img[:, :, np.newaxis], [1, 1, 3])
    bilateralled = cv2.bilateralFilter(np.array(rgb_img), -1, 20, 20)
    # equalized = cv2.equalizeHist(bilateralled[:, :, 0])
    # filtered = cv2.pyrMeanShiftFiltering(bilateralled, 5, 2)
    segments = slic(bilateralled, num_of_segment, convert2lab=True, compactness=compactness)
    f = mark_boundaries(rgb_img, segments, mode='subpixel')
    Image.fromarray(np.uint8(f * 255)).save('./cmc_slic/%d_%d.png' % (num_of_segment, int(compactness)))
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
    # standard_path = sorted(glob.glob("./AM002_20220812_3691639/standard/*/*.dcm"))
    # swift_path = sorted(glob.glob("./AM002_20220812_3691639/swift/*/*.dcm"))
    # swift_recon_low_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_low/*/*.dcm"))
    # swift_recon_medium_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_medium/*/*.dcm"))
    standard_path = sorted(glob.glob("./cmc_knee/standard/115/*/*.dcm"))
    recon_path = sorted(glob.glob("./cmc_knee/recon_M/115/*/*.dcm"))

    standard_img = norm_dcm_array((img_to_array(standard_path[n])))
    recon_img = norm_dcm_array((img_to_array(recon_path[n])))
    # swift_img = norm_dcm_array(img_to_array(swift_path[n]))
    # swift_recon_low_img = norm_dcm_array(img_to_array(swift_recon_low_path[n]))
    # swift_recon_medium_img = norm_dcm_array(img_to_array(swift_recon_medium_path[n]))
    if return_np_array:
        # return standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img
        return standard_img, recon_img

    standard_img = sitk.GetImageFromArray(standard_img)
    recon_img = sitk.GetImageFromArray(recon_img)
    # swift_img = sitk.GetImageFromArray(swift_img)
    # swift_recon_low_img = sitk.GetImageFromArray(swift_recon_low_img)
    # swift_recon_medium_img = sitk.GetImageFromArray(swift_recon_medium_img)

    # return standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img
    return standard_img, recon_img


def test_auto_seg(idx):
    # for m in range(50):
    # standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img = get_data(idx, True)
    standard_img, recon_img = get_data(idx, True)
    # imgs = [standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img]
    imgs = [standard_img, recon_img]
    result = []
    for index in range(len(imgs)):
        img = imgs[index]
        pred, segments = slic_mri(img, 64 * 64, 10)
        for i in range(pred.min(), pred.max() + 1):
            mask = np.zeros([512, 512, 3]).astype(np.uint8)
            for n in range(segments.min(), segments.max() + 1):
                if pred[n - 1] != i:
                    continue
                mask[np.where(segments == n)[0], np.where(segments == n)[1]] = [255, 0, 0]
            Image.blend(Image.fromarray(img).convert('RGBA'), Image.fromarray(mask).convert('RGBA'), 0.3).save(
                './cmc_seg/%d_mask_%d_%d.png' % (index, idx, i))
        result.append(mask)
    return result

def calculate_radiomics_features(img, mask):
    function_name_all = []
    result_all = []

    mask = mask[:, :, 0] / 255
    mask = sitk.GetImageFromArray(mask)

    # features = RadiomicsFirstOrder(img, mask)
    # features = RadiomicsGLCM(img, mask)
    # features = RadiomicsGLDM(img, mask)
    features = RadiomicsShape2D(img, mask)

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
            # print('%s Results: ' % function_name, result)
            function_name_all.append(function_name)
            result_all.append(result)
        except Exception as e:
            print("%s call failed" % function_name)
            print(e)
    return function_name_all, result_all

def extract_number(s):
    # Extract the number from a string
    if isinstance(s, str):
        number = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        return float(number[0]) if number else None
    return s

def calculate_std(csv_path):
    std_csv_path = "./cmc_results/Shape2D_std.csv" # output path

    df = pd.read_csv(csv_path)

    # Convert the values in the columns to numeric data types
    for column in ["Standard", "Recon"]:
        df[column] = df[column].apply(extract_number)

    # Compute the standard deviation for each function result
    function_names = df["Function"].unique()
    results_columns = ["Standard", "Recon"]

    # Store standard deviation metrics
    standard_deviations_data = []

    for function_name in function_names:
        function_data = df[df["Function"] == function_name]
        means = function_data[results_columns].mean()
        std_devs = function_data[results_columns].std()
        std_dev_percentages = (std_devs / means) * 100
        std_dev_and_percentages = tuple(
            [val for pair in zip(std_devs.tolist(), std_dev_percentages.tolist()) for val in pair]
        )
        standard_deviations_data.append((function_name,) + std_dev_and_percentages)

    # Create a new DataFrame with the standard deviation metrics
    columns = [
        "Function",
        "Std dev Standard",
        "Std dev % Standard",
        "Std dev Recon",
        "Std dev % Recon",
    ]
    standard_deviations_df = pd.DataFrame(standard_deviations_data, columns=columns)

    standard_deviations_df.to_csv(std_csv_path, index=False)

def analyze_radiomics():
    csv_path = "./cmc_results/Shape2D.csv"

    data = []
    for idx in range(10):
        # standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img = get_data(idx)
        standard_img, recon_img = get_data(idx)

        # temporary_mask = np.zeros([512, 512]).astype(np.uint8)
        # temporary_mask[200:250, 200:250] = 1

        result = test_auto_seg(idx)

        standard_function, standard_result = calculate_radiomics_features(standard_img, result[0])
        _, recon_result = calculate_radiomics_features(recon_img, result[1])
        # _, swift_result = calculate_radiomics_features(swift_img, result[1])
        # _, swift_recon_low_result = calculate_radiomics_features(swift_recon_low_img, result[2])
        # _, swift_recon_medium_result = calculate_radiomics_features(swift_recon_medium_img, result[3])

        # result_values = [standard_result, swift_result, swift_recon_low_result, swift_recon_medium_result]
        result_values = [standard_result, recon_result]

        for function_num, function_name in enumerate(standard_function):
            results = [result_values[i][function_num] for i in range(len(result_values))]
            row = [idx, function_name] + results
            data.append(row)

    # columns = ["Index", "Function", "Standard", "Swift", "Recon_L", "Recon_M"]
    columns = ["Index", "Function", "Standard", "Recon"]
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(csv_path, index=False)

    calculate_std(csv_path)

def main():
    # analyze_radiomics()
    calculate_std("./cmc_results/Shape2D.csv")


if __name__ == '__main__':
    main()