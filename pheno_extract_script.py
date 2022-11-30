from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from configparser import ConfigParser
import glob
from simba.rw_dfs import *


def calculate_convex_hull(x, y, p, threshold=0.2):
    if len(x) != len(p):
        raise ValueError('Got x and p with different lengths')

    selected_points, selected_p = [], []
    for i in range(len(x)):
        if p[i] > threshold:
            selected_points.append([x[i], y[i]])
            selected_p.append(p[i])

    if len(selected_points) < 3:
        return 0
    else:
        return ConvexHull(selected_points).volume


def calculate_weighted_avg(x, p=None, threshold=0.2):
    if p is not None and len(x) != len(p):
        raise ValueError('Got x and p with different lengths')

    selected_x, selected_p = [], []
    for i in range(len(x)):
        if p[i] > threshold:
            selected_x.append(x[i])
            selected_p.append(p[i])

    if len(selected_x) > 0:
        return np.ma.average(selected_x, weights=selected_p)
    else:
        return np.ma.average(x, weights=p)


def count_high_p(p, threshold=0.2):
    return len([1 for val in p if val > threshold])


def extract_features_userdef(inifile):
    print('Phenosimba says hello! This is version 9')
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    csv_dir_out = os.path.join(csv_dir, 'features_extracted')
    wfileType = config.get('General settings', 'workflow_file_type')
    vidInfPath = config.get('General settings', 'project_path')
    logsPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(logsPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    poseConfigPath = os.path.join(logsPath, 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    poseConfigDf = pd.read_csv(poseConfigPath, header=None)
    poseConfigDf = list(poseConfigDf[0])

    pup_threshold = 0.1
    dam_threshold = 0.5

    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)

    def count_values_in_range(series, values_in_range_min, values_in_range_max):
        return series.between(left=values_in_range_min, right=values_in_range_max).sum()

    roll_windows = []
    roll_windows_values = [1, 2, 5, 8, 15]

    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Extracting features from ' + str(len(filesFound)) + ' files...')

    for currentFile in filesFound:
        current_video_file = os.path.basename(currentFile.replace('.csv', ''))
        current_video_settings = vidinfDf.loc[vidinfDf['Video'] == current_video_file]
        try:
            pixels_per_mm = float(current_video_settings['pixels/mm'])
        except TypeError:
            print('Error: Cant find pixels/mm in video settings. Make sure all the videos that are going to be '
                  'analyzed are represented in the project_folder/logs/video_info.csv file')
            break
        fps = float(current_video_settings['fps'])
        print('Processing', str(current_video_file), '. Fps:', str(fps), '. mm/ppx:', str(pixels_per_mm))
        for j in range(len(roll_windows_values)):
            roll_windows.append(int(fps / roll_windows_values[j]))

        # Body parts will be in two categories: dam and pups
        body_part_names = list(poseConfigDf)
        dam_body_part_names, pup_body_part_names = [], []
        for bp in body_part_names:
            if 'pup' in bp:
                pup_body_part_names.append(bp)
            else:
                dam_body_part_names.append(bp)

        column_names = []
        column_names_shifted = []
        for bp in body_part_names:
            colHead1, colHead2, colHead3 = (bp + '_x', bp + '_y', bp + '_p')
            colHead4, colHead5, colHead6 = (bp + '_x_shifted', bp + '_y_shifted', bp + '_p_shifted')
            column_names.extend((colHead1, colHead2, colHead3))
            column_names_shifted.extend((colHead4, colHead5, colHead6))

        csv_df = read_df(currentFile, wfileType)
        csv_df.columns = column_names

        # csv_df = csv_df.fillna(0)
        csv_df = csv_df.drop(csv_df.index[[0]])
        csv_df = csv_df.apply(pd.to_numeric)
        csv_df = csv_df.reset_index(drop=True)

        # Calculate dam points
        print('Calculating dam points')

        csv_df['arm_x'] = np.where(csv_df['left_armpit_p'] > pup_threshold, csv_df['left_armpit_x'], csv_df['right_armpit_x'])
        csv_df['arm_y'] = np.where(csv_df['left_armpit_p'] > pup_threshold, csv_df['left_armpit_y'], csv_df['right_armpit_y'])
        csv_df['arm_p'] = np.where(csv_df['left_armpit_p'] > pup_threshold, csv_df['left_armpit_p'], csv_df['right_armpit_p'])

        csv_df['side_x'] = np.where(csv_df['left_ventrum_side_p'] > pup_threshold, csv_df['left_ventrum_side_x'], csv_df['right_ventrum_side_x'])
        csv_df['side_y'] = np.where(csv_df['left_ventrum_side_p'] > pup_threshold, csv_df['left_ventrum_side_y'], csv_df['right_ventrum_side_y'])
        csv_df['side_p'] = np.where(csv_df['left_ventrum_side_p'] > pup_threshold, csv_df['left_ventrum_side_p'], csv_df['right_ventrum_side_p'])

        # Calculate centroids
        print('Calculating dam centroids')
        dam_centroid_parts = ['dam_nose',
                              'left_eye',
                              'right_eye',
                              'left_ear',
                              'right_ear',
                              'left_shoulder',
                              'right_shoulder',
                              'left_armpit',
                              'right_armpit',
                              'left_ventrum_side',
                              'right_ventrum_side',
                              'back_2',
                              'back_4',
                              'back_8',
                              'back_10']
        csv_df['dam_centroid_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                    [row[str(column) + '_x'] for column in dam_centroid_parts],
                                                    [row[str(column) + '_p'] for column in dam_centroid_parts]), axis=1)
        csv_df['dam_centroid_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                    [row[str(column) + '_y'] for column in dam_centroid_parts],
                                                    [row[str(column) + '_p'] for column in dam_centroid_parts]), axis=1)

        dam_head_parts = ['dam_nose',
                          'left_eye',
                          'right_eye',
                          'left_ear',
                          'right_ear']
        csv_df['head_centroid_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                    [row[str(column) + '_x'] for column in dam_head_parts],
                                                    [row[str(column) + '_p'] for column in dam_head_parts]), axis=1)
        csv_df['head_centroid_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                    [row[str(column) + '_y'] for column in dam_head_parts],
                                                    [row[str(column) + '_p'] for column in dam_head_parts]), axis=1)

        print('Calculating single points per pup')
        for pup in range(8):
            pup_name = 'pup' + str(pup+1)
            csv_df[pup_name + '_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                                                [row[pup_name + '_nose_x'],
                                                                                 row[pup_name + '_eyes_x'],
                                                                                 row[pup_name + '_ears_x'],
                                                                                 row[pup_name + '_back1_x'],
                                                                                 row[pup_name + '_back2_x'],
                                                                                 row[pup_name + '_back3_x'],
                                                                                 row[pup_name + '_back4_x'],
                                                                                 row[pup_name + '_back5_x'],
                                                                                 row[pup_name + '_back6_x']],
                                                                                [row[pup_name + '_nose_p'],
                                                                                 row[pup_name + '_eyes_p'],
                                                                                 row[pup_name + '_ears_p'],
                                                                                 row[pup_name + '_back1_p'],
                                                                                 row[pup_name + '_back2_p'],
                                                                                 row[pup_name + '_back3_p'],
                                                                                 row[pup_name + '_back4_p'],
                                                                                 row[pup_name + '_back5_p'],
                                                                                 row[pup_name + '_back6_p']]), axis=1)
            csv_df[pup_name + '_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                                                [row[pup_name + '_nose_y'],
                                                                                 row[pup_name + '_eyes_y'],
                                                                                 row[pup_name + '_ears_y'],
                                                                                 row[pup_name + '_back1_y'],
                                                                                 row[pup_name + '_back2_y'],
                                                                                 row[pup_name + '_back3_y'],
                                                                                 row[pup_name + '_back4_y'],
                                                                                 row[pup_name + '_back5_y'],
                                                                                 row[pup_name + '_back6_y']],
                                                                                [row[pup_name + '_nose_p'],
                                                                                 row[pup_name + '_eyes_p'],
                                                                                 row[pup_name + '_ears_p'],
                                                                                 row[pup_name + '_back1_p'],
                                                                                 row[pup_name + '_back2_p'],
                                                                                 row[pup_name + '_back3_p'],
                                                                                 row[pup_name + '_back4_p'],
                                                                                 row[pup_name + '_back5_p'],
                                                                                 row[pup_name + '_back6_p']]), axis=1)

            csv_df[pup_name + '_p'] = np.ma.average([csv_df[pup_name + '_nose_y'],
                                                     csv_df[pup_name + '_eyes_y'],
                                                     csv_df[pup_name + '_ears_y'],
                                                     csv_df[pup_name + '_back1_y'],
                                                     csv_df[pup_name + '_back2_y'],
                                                     csv_df[pup_name + '_back3_y'],
                                                     csv_df[pup_name + '_back4_y'],
                                                     csv_df[pup_name + '_back5_y'],
                                                     csv_df[pup_name + '_back6_y']], axis=0)

        print('Calculating pups centroid')

        pups = ['pup' + str(i+1) for i in range(8)]
        csv_df['pups_centroid_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                                                [row[p + '_x'] for p in pups],
                                                                                [row[p + '_p'] for p in pups]), axis=1)
        csv_df['pups_centroid_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                                                [row[p + '_y'] for p in pups],
                                                                                [row[p + '_p'] for p in pups]), axis=1)

        # Go through every row individually
        print('Calculating pup convex hull and high probably body part counts')
        csv_df['pups_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                                                            [row['pup' + str(i+1) + '_x'] for i in range(8)],
                                                            [row['pup' + str(i+1) + '_y'] for i in range(8)],
                                                            [row['pup' + str(i+1) + '_p'] for i in range(8)]), axis=1)

        csv_df['pup_avg_p'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                            [row['pup' + str(i+1) + '_x'] for i in range(8)]
                                                            ), axis=1)

        csv_df['high_p_pups'] = csv_df.apply(lambda row: count_high_p(
                                                            [row['pup' + str(i+1) + '_p'] for i in range(8)]
                                                            ), axis=1)
        csv_df['high_p_dam_bp'] = csv_df.apply(lambda row: count_high_p(
                                                            [row[bp + '_p'] for bp in dam_body_part_names]
                                                            ), axis=1)


        # Create a shifted dataframe for distance calculations
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = column_names_shifted
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined = csv_df_combined.reset_index(drop=True)

        # Calculate movements
        print('Calculating movements')
        movement_columns = dam_body_part_names
        movement_columns += ['pup' + str(i+1) for i in range(8)]

        for bp in movement_columns:
            column_name = 'movement_' + bp
            bpX_1, bpY_1 = (bp + '_x', bp + '_y')
            bpX_2, bpY_2 = (bp + '_x_shifted', bp + '_y_shifted')
            csv_df[column_name] = (np.sqrt((csv_df_combined[bpX_1] - csv_df_combined[bpX_2]) ** 2 + (csv_df_combined[bpY_1] - csv_df_combined[bpY_2]) ** 2)) / pixels_per_mm

        csv_df['back_avg_movement'] = np.ma.average([csv_df['movement_back_2'],
                                                     csv_df['movement_back_4'],
                                                     csv_df['movement_back_8'],
                                                     csv_df['movement_back_10']],
                                              axis=0)
        csv_df['head_avg_movement'] = np.ma.average([csv_df['movement_dam_nose'],
                                                     csv_df['movement_right_eye'],
                                                     csv_df['movement_left_ear'],
                                                     csv_df['movement_right_ear']],
                                              axis=0)
        csv_df['head_max_movement'] = np.ma.max([csv_df['movement_dam_nose'],
                                                     csv_df['movement_right_eye'],
                                                     csv_df['movement_left_ear'],
                                                     csv_df['movement_right_ear']],
                                              axis=0)

        csv_df['sum_pup_movement'] = csv_df['movement_pup1'] + csv_df['movement_pup2'] + csv_df['movement_pup3'] \
                                    + csv_df['movement_pup4'] + csv_df['movement_pup5'] + csv_df['movement_pup6'] \
                                    + csv_df['movement_pup7']  + csv_df['movement_pup8']

        # Moving average of movement
        print('Calculating moving average of movements')
        csv_df['head_avg_movement_mavg_30'] = csv_df['head_avg_movement'].rolling(roll_windows[0], min_periods=1).mean()
        csv_df['head_avg_movement_mavg_6'] = csv_df['head_avg_movement'].rolling(roll_windows[2], min_periods=1).mean()
        csv_df['head_avg_movement_mavg_2'] = csv_df['head_avg_movement'].rolling(roll_windows[4], min_periods=1).mean()

        csv_df['head_max_movement_mavg_30'] = csv_df['head_max_movement'].rolling(roll_windows[0], min_periods=1).mean()
        csv_df['head_max_movement_mavg_6'] = csv_df['head_max_movement'].rolling(roll_windows[2], min_periods=1).mean()
        csv_df['head_max_movement_mavg_2'] = csv_df['head_max_movement'].rolling(roll_windows[4], min_periods=1).mean()

        csv_df['back_avg_movement_mavg_30'] = csv_df['back_avg_movement'].rolling(roll_windows[0], min_periods=1).mean()
        csv_df['back_avg_movement_mavg_6'] = csv_df['back_avg_movement'].rolling(roll_windows[2], min_periods=1).mean()
        csv_df['back_avg_movement_mavg_2'] = csv_df['back_avg_movement'].rolling(roll_windows[4], min_periods=1).mean()

        csv_df['head_back_rel_mov_30'] = csv_df['head_avg_movement_mavg_30'] / (csv_df['head_avg_movement_mavg_30'] + csv_df['back_avg_movement_mavg_30'])
        csv_df['head_back_rel_mov_6'] = csv_df['head_avg_movement_mavg_6'] / (csv_df['head_avg_movement_mavg_6'] + csv_df['back_avg_movement_mavg_6'])
        csv_df['head_back_rel_mov_2'] = csv_df['head_avg_movement_mavg_2'] / (csv_df['head_avg_movement_mavg_2'] + csv_df['back_avg_movement_mavg_2'])

        # Distance calculations
        print('Calculating distances')
        csv_df['dam_pup_distance'] = np.sqrt((csv_df['dam_centroid_x'] - csv_df['pups_centroid_x'])**2 + (csv_df['dam_centroid_y'] - csv_df['pups_centroid_y'])**2)
        csv_df['head_pup_distance'] = np.sqrt((csv_df['head_centroid_x'] - csv_df['pups_centroid_x'])**2 + (csv_df['head_centroid_y'] - csv_df['pups_centroid_y'])**2)

        csv_df['back_length'] = np.sqrt((csv_df['back_2_x'] - csv_df['back_10_x'])**2
                                             + (csv_df['back_2_y'] - csv_df['back_10_y'])**2)

        csv_df['nose_back10_length'] = np.sqrt((csv_df['dam_nose_x'] - csv_df['back_10_x'])**2
                                             + (csv_df['dam_nose_y'] - csv_df['back_10_y'])**2)
        csv_df['nose_back2_length'] = np.sqrt((csv_df['dam_nose_x'] - csv_df['back_2_x'])**2
                                             + (csv_df['dam_nose_y'] - csv_df['back_2_y'])**2)

        csv_df['avg_dam_bp_p'] = np.ma.average([ csv_df['dam_nose_p'],
                                                 csv_df['left_eye_p'],
                                                 csv_df['right_eye_p'],
                                                 csv_df['left_ear_p'],
                                                 csv_df['right_ear_p'],
                                                 csv_df['left_shoulder_p'],
                                                 csv_df['right_shoulder_p'],
                                                 csv_df['arm_p'],
                                                 csv_df['side_p']],
                                              axis=0)

        csv_df['sum_probabilities'] = csv_df[[
                                                'dam_nose_p',
                                                'left_eye_p',
                                                'right_eye_p',
                                                'left_ear_p',
                                                'right_ear_p',
                                                'left_shoulder_p',
                                                'right_shoulder_p',
                                                'arm_p',
                                                'side_p',
                                                'pup1_p',
                                                'pup2_p',
                                                'pup3_p',
                                                'pup4_p',
                                                'pup5_p',
                                                'pup6_p',
                                                'pup7_p',
                                                'pup8_p']].sum(axis=1)

        # Drop columns
        csv_df = csv_df.drop([
            'movement_back_2',
            'movement_back_4',
            'movement_back_8',
            'movement_back_10',
            'movement_pup1',
            'movement_pup2',
            'movement_pup3',
            'movement_pup4',
            'movement_pup5',
            'movement_pup6',
            'movement_pup7',
            'movement_pup8'
        ], axis=1)

        # Save DF
        print('Exporting df')
        #csv_df = csv_df.loc[:, ~csv_df.T.duplicated(keep='first')]
        csv_df = csv_df.reset_index(drop=True)
        csv_df = csv_df.fillna(0)
        #csv_df = csv_df.drop(columns=['index'])
        fileOutName = os.path.basename(currentFile).replace('.' + wfileType, '')
        savePath = os.path.join(csv_dir_out, fileOutName) + '.csv'
        print('Save path:', savePath)
        save_df(csv_df, wfileType, savePath)
        print('Feature extraction complete for ' + '"' + str(current_video_file) + '".')

    print('All feature extraction complete.')
