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
from scipy.stats import pearsonr
from PIL import Image
from utils import norm_dcm_array, img_to_array
import radiomics
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from radiomics.firstorder import RadiomicsFirstOrder
from radiomics.shape2D import RadiomicsShape2D
from radiomics.glcm import RadiomicsGLCM
from radiomics.glrlm import RadiomicsGLRLM
from radiomics.glszm import RadiomicsGLSZM
from radiomics.ngtdm import RadiomicsNGTDM
from radiomics.gldm import RadiomicsGLDM
import SimpleITK as sitk
import cv2
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
import pandas as pd
import re
from tqdm import tqdm
import copy
import os
import tempfile
from sklearn.linear_model import LinearRegression

DATA_TYPE = 'yuhs'


def slic_mri(np_img: np.ndarray, num_of_segment: int, compactness: float, num_clusters: int = 16):
    height, width = np_img.shape
    rgb_img = np.tile(np_img[:, :, np.newaxis], [1, 1, 3])
    bilateralled = cv2.bilateralFilter(np.array(rgb_img), -1, 20, 20)
    # equalized = cv2.equalizeHist(bilateralled[:, :, 0])
    # filtered = cv2.pyrMeanShiftFiltering(bilateralled, 5, 2)
    segments = slic(bilateralled, num_of_segment, convert2lab=True, compactness=compactness)
    f = mark_boundaries(rgb_img, segments, mode='subpixel')
    # Image.fromarray(np.uint8(f * 255)).save('./%s_slic/%d_%d.png' % (DATA_TYPE, num_of_segment, int(compactness)))
    mean_list = []
    std_list = []
    x_list = []
    y_list = []
    for n in range(segments.min(), segments.max() + 1):
        mean_val = np.mean(np_img[np.where(segments == n)[0], np.where(segments == n)[1]] / 255.)
        std_val = np.std(np_img[np.where(segments == n)[0], np.where(segments == n)[1]] / 255.)
        y_val = (np.mean(np.where(segments == n)[0]) - height // 2) / float(height)
        x_val = (np.mean(np.where(segments == n)[1]) - width // 2) / float(width)
        # if mean_val < 10:
        #     continue
        mean_list.append(mean_val)
        std_list.append(std_val)
        y_list.append(y_val)
        x_list.append(x_val)
    norm_mean = 2 * np.array(mean_list)  # / np.std(mean_list)
    norm_std = 0.5 * np.array(std_list)  # / np.std(std_list)
    norm_y = 0.2 * np.array(y_list)  # / np.std(y_list)
    norm_x = 0.2 * np.array(x_list)  # / np.std(x_list)
    # cluster = DBSCAN()
    cluster = SpectralClustering(num_clusters)
    pred = cluster.fit_predict(np.concatenate(
        [norm_mean[:, np.newaxis], norm_std[:, np.newaxis], norm_y[:, np.newaxis], norm_x[:, np.newaxis]], axis=1))
    # dbscan.fit_predict()
    return pred, segments


def test_auto_seg(np_img: np.ndarray, save_path: str = None):
    # for m in range(50):
    height, width = np_img.shape
    # standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img = get_data(idx, True)
    # imgs = [standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img]
    pred, segments = slic_mri(np_img, (height // 16) * (width // 16), 10)
    masks = []
    for i in range(pred.min(), pred.max() + 1):
        mask = np.zeros([height, width, 3]).astype(np.uint8)
        for n in range(segments.min(), segments.max() + 1):
            if pred[n - 1] != i:
                continue
            mask[np.where(segments == n)[0], np.where(segments == n)[1]] = [255, 0, 0]
        masks.append(mask)

    ''' Here, you should find meaningful mask among all masks '''
    final_mask = np.zeros_like(np_img)
    for i in range(len(masks)):
        # print(np.mean(np_img * (masks[i][:, :, 0] / 255)))
        if np.mean(np_img * (masks[i][:, :, 0] / 255)) > 1:
            final_mask += masks[i][:, :, 0]
            # Image.blend(Image.fromarray(np_img).convert('RGBA'), Image.fromarray(masks[i]).convert('RGBA'), 0.3).save(
            #     'A_%d.png' % i)
        # else:
        #     Image.blend(Image.fromarray(np_img).convert('RGBA'), Image.fromarray(masks[i]).convert('RGBA'), 0.3).save(
        #         'B_%d.png' % i)

    if save_path is not None:
        Image.blend(Image.fromarray(np_img).convert('RGBA'), Image.fromarray(final_mask).convert('RGBA'), 0.3).save(
            save_path)
    return final_mask


def get_data(n):
    # standard_path = sorted(glob.glob("./AM002_20220812_3691639/standard/*/*.dcm"))
    # swift_path = sorted(glob.glob("./AM002_20220812_3691639/swift/*/*.dcm"))
    # swift_recon_low_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_low/*/*.dcm"))
    # swift_recon_medium_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_medium/*/*.dcm"))
    standard_path = sorted(glob.glob("./%s_knee_cor_bone_marrow/standard/*.dcm" % DATA_TYPE))
    swift_path = sorted(glob.glob("./%s_knee_cor_bone_marrow/swift/*.dcm" % DATA_TYPE))
    recon_low_path = sorted(glob.glob("./%s_knee_cor_bone_marrow/recon_low/*.dcm" % DATA_TYPE))
    recon_med_path = sorted(glob.glob("./%s_knee_cor_bone_marrow/recon_med/*.dcm" % DATA_TYPE))

    standard_img_npy = norm_dcm_array((img_to_array(standard_path[n])))
    swift_img_npy = norm_dcm_array((img_to_array(swift_path[n])))
    recon_low_img_npy = norm_dcm_array((img_to_array(recon_low_path[n])))
    recon_med_img_npy = norm_dcm_array((img_to_array(recon_med_path[n])))
    # swift_img = norm_dcm_array(img_to_array(swift_path[n]))
    # swift_recon_low_img = norm_dcm_array(img_to_array(swift_recon_low_path[n]))
    # swift_recon_medium_img = norm_dcm_array(img_to_array(swift_recon_medium_path[n]))

    standard_img = sitk.GetImageFromArray(standard_img_npy)
    swift_img = sitk.GetImageFromArray(swift_img_npy)
    recon_low_img = sitk.GetImageFromArray(recon_low_img_npy)
    recon_med_img = sitk.GetImageFromArray(recon_med_img_npy)
    # swift_img = sitk.GetImageFromArray(swift_img)
    # swift_recon_low_img = sitk.GetImageFromArray(swift_recon_low_img)
    # swift_recon_medium_img = sitk.GetImageFromArray(swift_recon_medium_img)

    # return standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img
    return standard_img, swift_img, recon_low_img, recon_med_img, standard_img_npy, swift_img_npy, recon_low_img_npy, recon_med_img_npy


def calculate_radiomics_features(img, mask, class_features):
    function_name_all = []
    result_all = []

    mask = mask // 255
    mask = sitk.GetImageFromArray(mask)

    features = class_features(img, mask)
    if type(features) in [RadiomicsFirstOrder, RadiomicsGLCM, RadiomicsGLRLM, RadiomicsGLSZM, RadiomicsNGTDM,
                          RadiomicsGLDM]:
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
    std_csv_path = csv_path.replace('.csv', '_std.csv')  # output path

    df = pd.read_csv(csv_path)

    # Convert the values in the columns to numeric data types
    for column in ["Standard", "Swift", "Recon_L", "Recon_M"]:
        df[column] = df[column].apply(extract_number)

    # Compute the standard deviation for each function result
    function_names = df["Function"].unique()
    results_columns = ["Standard", "Swift", "Recon_L", "Recon_M"]

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
        "Std dev Swift",
        "Std dev % Swift",
        "Std dev Recon_L",
        "Std dev % Recon_L",
        "Std dev Recon_M",
        "Std dev % Recon_M",
    ]
    standard_deviations_df = pd.DataFrame(standard_deviations_data, columns=columns)

    standard_deviations_df.to_csv(std_csv_path, index=False)


def on_mouse(event, x, y, flags, param):  # (x,y)의 (0,0)은 좌측 상단점 기준
    global final_img, cache, point1, tissue_1_radius

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 좌클릭 시 빨간색 원 그려짐
        print(x, y)
        cache = copy.deepcopy(final_img)
        point1.append((x, y))
        cv2.circle(final_img, (x, y), tissue_1_radius, color=(0, 0, 255), thickness=-1)  # roi mask 1 생성
        cv2.imshow('window', final_img)


def get_roi(img_npy: np.ndarray, save_path: str = None, x_center: int = 0, y_center: int = 0, roi_exist: bool = False):
    global final_img, cache, point1, tissue_1_radius

    tissue_1_radius = img_npy.shape[0] // 16

    if roi_exist:
        tissue_x_center1 = x_center
        tissue_y_center1 = y_center

    else:

        pil_img = Image.fromarray(img_npy)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
            pil_img.save(temp_file.name)
            final_img = cv2.imread(temp_file.name)

        point1 = []

        cv2.namedWindow('window')
        cv2.setMouseCallback('window', on_mouse, final_img)
        cv2.imshow('window', final_img)
        while True:
            if cv2.waitKey(0) & 0xFF == 27:  # esc로 종료
                print('ESC Pressed, the windows will closed')
                cv2.destroyAllWindows()
                break
            elif cv2.waitKey(0) & 0xFF == 32:  # spacebar 두 번 누르면 undo
                final_img = copy.deepcopy(cache)
                cv2.imshow('window', final_img)
            else:
                continue

        tissue_x_center1 = point1[0][0]
        tissue_y_center1 = point1[0][1]
        # tissue_point1 = list(set([(int(tissue_x_center1 + tissue_1_radius * np.cos(theta)),
        #                            int(tissue_y_center1 + tissue_1_radius * np.sin(theta)))
        #                           for theta in np.linspace(0, 2*np.pi, 360)]))

        # tissue_pixel1 = []

    mask = np.zeros_like((img_npy)).astype(np.uint8)
    mask = cv2.circle(mask, (tissue_x_center1, tissue_y_center1), radius=tissue_1_radius, color=(255, 0, 0),
                      thickness=-1)

    # for i in range(0, len(tissue_point1)):
    #     tissue_pixel1.append(img_npy[tissue_point1[i]])

    if save_path is not None:
        Image.blend(Image.fromarray(img_npy).convert('RGBA'), Image.fromarray(mask).convert('RGBA'), 0.3).save(
            save_path)

    return mask, tissue_x_center1, tissue_y_center1


def analyze_radiomics():
    standard_x, standard_y, swift_x, swift_y, recon_low_x, recon_low_y, recon_med_x, recon_med_y = [], [], [], [], [], [], [], []
    feature_list = [RadiomicsFirstOrder, RadiomicsShape2D, RadiomicsGLCM, RadiomicsGLRLM, RadiomicsGLSZM,
                    RadiomicsNGTDM, RadiomicsGLDM]
    for feature in feature_list:
        data = []
        feature_name = str(feature).split("'")[1].split('.')[-1]
        csv_path = "./%s_radiomics/%s.csv" % (DATA_TYPE, feature_name)
        for idx in tqdm(range(45)):
            # standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img = get_data(idx)
            standard_img, swift_img, recon_low_img, recon_med_img, standard_img_npy, swift_img_npy, recon_low_img_npy, recon_med_img_npy = get_data(
                idx)
            # temporary_mask = np.zeros([512, 512]).astype(np.uint8)
            # temporary_mask[200:250, 200:250] = 1

            try:
                standard_img_mask, _, _ = get_roi(standard_img_npy,
                                                  './%s_roi/%03d_standard_mask.png' % (DATA_TYPE, idx), standard_x[idx],
                                                  standard_y[idx], True)
                swift_img_mask, _, _ = get_roi(swift_img_npy, './%s_roi/%03d_swift_mask.png' % (DATA_TYPE, idx),
                                               swift_x[idx], swift_y[idx], True)
                recon_low_img_mask, _, _ = get_roi(recon_low_img_npy,
                                                   './%s_roi/%03d_recon_low_mask.png' % (DATA_TYPE, idx),
                                                   recon_low_x[idx], recon_low_y[idx], True)
                recon_med_img_mask, _, _ = get_roi(recon_med_img_npy,
                                                   './%s_roi/%03d_recon_med_mask.png' % (DATA_TYPE, idx),
                                                   recon_med_x[idx], recon_med_y[idx], True)

            except:
                standard_img_mask, st_x, st_y = get_roi(standard_img_npy,
                                                        './%s_roi/%03d_standard_mask.png' % (DATA_TYPE, idx), 0, 0,
                                                        False)
                standard_x.append(st_x)
                standard_y.append(st_y)
                swift_img_mask, sw_x, sw_y = get_roi(swift_img_npy, './%s_roi/%03d_swift_mask.png' % (DATA_TYPE, idx),
                                                     0, 0, False)
                swift_x.append(sw_x)
                swift_y.append(sw_y)
                recon_low_img_mask, rl_x, rl_y = get_roi(recon_low_img_npy,
                                                         './%s_roi/%03d_recon_low_mask.png' % (DATA_TYPE, idx), 0, 0,
                                                         False)
                recon_low_x.append(rl_x)
                recon_low_y.append(rl_y)
                recon_med_img_mask, rm_x, rm_y = get_roi(recon_med_img_npy,
                                                         './%s_roi/%03d_recon_med_mask.png' % (DATA_TYPE, idx), 0, 0,
                                                         False)
                recon_med_x.append(rm_x)
                recon_med_y.append(rm_y)

            # standard_img_mask = test_auto_seg(standard_img_npy, '%s_seg/%03d_standard_mask.png' % (DATA_TYPE, idx))
            # recon_img_mask = test_auto_seg(recon_img_npy, '%s_seg/%03d_recon_mask.png' % (DATA_TYPE, idx))

            standard_function, standard_result = calculate_radiomics_features(standard_img, standard_img_mask, feature)
            _, swift_result = calculate_radiomics_features(swift_img, swift_img_mask, feature)
            _, recon_low_result = calculate_radiomics_features(recon_low_img, recon_low_img_mask, feature)
            _, recon_med_result = calculate_radiomics_features(recon_med_img, recon_med_img_mask, feature)
            # _, swift_result = calculate_radiomics_features(swift_img, result[1])
            # _, swift_recon_low_result = calculate_radiomics_features(swift_recon_low_img, result[2])
            # _, swift_recon_medium_result = calculate_radiomics_features(swift_recon_medium_img, result[3])

            # result_values = [standard_result, swift_result, swift_recon_low_result, swift_recon_medium_result]
            result_values = [standard_result, swift_result, recon_low_result, recon_med_result]

            for function_num, function_name in enumerate(standard_function):
                results = [result_values[i][function_num] for i in range(len(result_values))]
                row = [idx, function_name] + results
                data.append(row)

        columns = ["Index", "Function", "Standard", "Swift", "Recon_L", "Recon_M"]
        df = pd.DataFrame(data, columns=columns)

        df.to_csv(csv_path, index=False)

        calculate_std(csv_path)


def regression_test():
    ''' Parsing csv '''
    subject_num = 45
    radiomics_feature_list = [('RadiomicsFirstOrder', 19), ('RadiomicsGLCM', 24), ('RadiomicsGLRLM', 16),
                              ('RadiomicsGLSZM', 16), ('RadiomicsNGTDM', 5), ('RadiomicsGLDM', 14)]
    standard_arrays = []
    swift_arrays = []
    recon_l_arrays = []
    recon_m_arrays = []
    for radiomics_feature, feature_num in radiomics_feature_list:
        standard_features = []
        swift_features = []
        recon_l_features = []
        recon_m_features = []
        df = pd.read_csv('results/%s.csv' % radiomics_feature)
        for sub_idx in range(subject_num):
            tmp_standard_features = []
            tmp_swift_features = []
            tmp_recon_l_features = []
            tmp_recon_m_features = []
            for feature_idx in range(feature_num):
                features = df.iloc[sub_idx * feature_num + feature_idx]
                if len(features['Standard']) == 1:
                    standard_feature = 0.
                else:
                    standard_feature = float(features['Standard'][1:-1])
                if len(features['Swift']) == 1:
                    swift_feature = 0.
                else:
                    swift_feature = float(features['Swift'][1:-1])
                if len(features['Recon_L']) == 1:
                    recon_l_feature = 0.
                else:
                    recon_l_feature = float(features['Recon_L'][1:-1])
                if len(features['Recon_M']) == 1:
                    recon_m_feature = 0.
                else:
                    recon_m_feature = float(features['Recon_M'][1:-1])
                tmp_standard_features.append(standard_feature)
                tmp_swift_features.append(swift_feature)
                tmp_recon_l_features.append(recon_l_feature)
                tmp_recon_m_features.append(recon_m_feature)
            standard_features.append(tmp_standard_features)
            swift_features.append(tmp_swift_features)
            recon_l_features.append(tmp_recon_l_features)
            recon_m_features.append(tmp_recon_m_features)

        standard_arrays.append(standard_features)
        swift_arrays.append(swift_features)
        recon_l_arrays.append(recon_l_features)
        recon_m_arrays.append(recon_m_features)
    standard_x = np.concatenate(standard_arrays, axis=1)
    swift_x = np.concatenate(swift_arrays, axis=1)
    recon_l_x = np.concatenate(recon_l_arrays, axis=1)
    recon_m_x = np.concatenate(recon_m_arrays, axis=1)
    total_x = np.concatenate([standard_x, swift_x, recon_l_x, recon_m_x], axis=0)
    # normalize each x
    standard_x = (standard_x - np.mean(total_x, axis=0)) / np.std(total_x, axis=0)
    swift_x = (swift_x - np.mean(total_x, axis=0)) / np.std(total_x, axis=0)
    recon_l_x = (recon_l_x - np.mean(total_x, axis=0)) / np.std(total_x, axis=0)
    recon_m_x = (recon_m_x - np.mean(total_x, axis=0)) / np.std(total_x, axis=0)

    df = pd.read_excel('./results/age_gt.xlsx')
    y = np.array(list(df.iloc[2:47]['Unnamed: 3']))

    standard_pearson = []
    swift_pearson = []
    recon_l_pearson = []
    recon_m_pearson = []
    for i in range(standard_x.shape[1]):
        standard_pearson.append(abs(pearsonr(standard_x[:, i], y)[0]))
        swift_pearson.append(abs(pearsonr(swift_x[:, i], y)[0]))
        recon_l_pearson.append(abs(pearsonr(recon_l_x[:, i], y)[0]))
        recon_m_pearson.append(abs(pearsonr(recon_m_x[:, i], y)[0]))

    standard_pearson = np.array(standard_pearson)
    swift_pearson = np.array(swift_pearson)
    recon_l_pearson = np.array(recon_l_pearson)
    recon_m_pearson = np.array(recon_m_pearson)
    total_pearson = standard_pearson + swift_pearson + recon_l_pearson + recon_m_pearson
    pearson_arg_sort = np.argsort(total_pearson)[::-1]

    for selected_feature_num in range(10, 20):
        # selected_feature_num = 10
        print('--------------------------------------------------')
        print('Selected feature nums is %d' % selected_feature_num)
        selected_standard_x = standard_x[:, pearson_arg_sort[:selected_feature_num]]
        selected_swift_x = swift_x[:, pearson_arg_sort[:selected_feature_num]]
        selected_recon_l_x = recon_l_x[:, pearson_arg_sort[:selected_feature_num]]
        selected_recon_m_x = recon_m_x[:, pearson_arg_sort[:selected_feature_num]]

        reg_standard = LinearRegression().fit(selected_standard_x, y)
        print('standard: ', reg_standard.score(selected_standard_x, y))
        reg_swift = LinearRegression().fit(selected_swift_x, y)
        print('swift: ', reg_swift.score(selected_swift_x, y))
        reg_recon_l = LinearRegression().fit(selected_recon_l_x, y)
        print('recon_l: ', reg_recon_l.score(selected_recon_l_x, y))
        reg_recon_m = LinearRegression().fit(selected_recon_m_x, y)
        print('recon_m: ', reg_recon_m.score(selected_recon_m_x, y))

        y_standard = reg_standard.predict(selected_standard_x)
        y_swift = reg_swift.predict(selected_swift_x)
        y_recon_l = reg_recon_l.predict(selected_recon_l_x)
        y_recon_m = reg_recon_m.predict(selected_recon_m_x)

        print('mse_standard: ', mean_squared_error(y, y_standard))
        print('mse_swift: ', mean_squared_error(y, y_swift))
        print('mse_recon_l: ', mean_squared_error(y, y_recon_l))
        print('mse_recon_m: ', mean_squared_error(y, y_recon_m))

        n = 5
        top_n_error_index = np.argsort(np.abs(y - y_recon_l))[::-1][:n]

        fig = plt.figure()
        plt.scatter(np.ones(45), np.sqrt(np.square(y - y_standard)), alpha=0.2, color='black')
        plt.scatter(np.ones(n), np.sqrt(np.square(y - y_standard))[top_n_error_index], alpha=0.5, color='red')
        plt.scatter(1, np.mean(np.sqrt(np.square(y - y_standard))), alpha=1., marker='*', color='red', edgecolors='red')
        plt.plot(np.ones(2), [np.mean(np.sqrt(np.square(y - y_standard))) - np.std(np.sqrt(np.square(y - y_standard))),
                              np.mean(np.sqrt(np.square(y - y_standard))) + np.std(np.sqrt(np.square(y - y_standard)))],
                 color='red')

        plt.scatter(np.ones(45) * 2, np.sqrt(np.square(y - y_swift)), alpha=0.2, color='yellow')
        plt.scatter(np.ones(n) * 2, np.sqrt(np.square(y - y_swift))[top_n_error_index], alpha=0.5, color='red')
        plt.scatter(2, np.mean(np.sqrt(np.square(y - y_swift))), alpha=1., marker='*', color='red', edgecolors='black')
        plt.plot(np.ones(2) * 2, [np.mean(np.sqrt(np.square(y - y_swift))) - np.std(np.sqrt(np.square(y - y_swift))),
                                  np.mean(np.sqrt(np.square(y - y_swift))) + np.std(np.sqrt(np.square(y - y_swift)))],
                 color='red')

        plt.scatter(np.ones(45) * 3, np.sqrt(np.square(y - y_recon_l)), alpha=0.2, color='blue')
        plt.scatter(np.ones(n) * 3, np.sqrt(np.square(y - y_recon_l))[top_n_error_index], alpha=0.5, color='red')
        plt.scatter(3, np.mean(np.sqrt(np.square(y - y_recon_l))), alpha=1., marker='*', color='red',
                    edgecolors='black')
        plt.plot(np.ones(2) * 3,
                 [np.mean(np.sqrt(np.square(y - y_recon_l))) - np.std(np.sqrt(np.square(y - y_recon_l))),
                  np.mean(np.sqrt(np.square(y - y_recon_l))) + np.std(np.sqrt(np.square(y - y_recon_l)))],
                 color='red')

        plt.scatter(np.ones(45) * 4, np.sqrt(np.square(y - y_recon_m)), alpha=0.2, color='green')
        plt.scatter(np.ones(n) * 4, np.sqrt(np.square(y - y_recon_m))[top_n_error_index], alpha=0.5, color='red')
        plt.scatter(4, np.mean(np.sqrt(np.square(y - y_recon_m))), alpha=1., marker='*', color='red',
                    edgecolors='black')
        plt.plot(np.ones(2) * 4,
                 [np.mean(np.sqrt(np.square(y - y_recon_m))) - np.std(np.sqrt(np.square(y - y_recon_m))),
                  np.mean(np.sqrt(np.square(y - y_recon_m))) + np.std(np.sqrt(np.square(y - y_recon_m)))],
                 color='red')
        fig.savefig('./scatter_%d.png' % selected_feature_num)
        import scipy.stats as stats
        t_stat, p_val = stats.ttest_ind(np.sqrt(np.square(y - y_standard)), np.sqrt(np.square(y - y_recon_l)),
                                        equal_var=False)
        print('p-value with recon_l is ', p_val)
        t_stat, p_val = stats.ttest_ind(np.sqrt(np.square(y - y_standard)), np.sqrt(np.square(y - y_recon_l)),
                                        equal_var=False)
        print('p-value with recon_m is ', p_val)


def main():
    regression_test()
    # analyze_radiomics()


if __name__ == '__main__':
    main()
