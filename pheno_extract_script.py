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
    if p is not None:
        p = [0 if val is None else val for val in p]
        for i in range(len(x)):
            if p[i] > threshold:
                selected_x.append(x[i])
                selected_p.append(p[i])

    if len(selected_x) > 0:
        return np.ma.average(selected_x, weights=selected_p)
    else:
        return np.ma.average(x)


def count_high_p(p, threshold=0.2):
    return len([1 for val in p if val > threshold])


def extract_features_userdef(inifile):
    print('Phenosimba says hello! This is version 9.2')
    config = ConfigParser()
    config_file = str(inifile)
    config.read(config_file)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    csv_dir_out = os.path.join(csv_dir, 'features_extracted')
    workflow_type = config.get('General settings', 'workflow_file_type')
    info_path = config.get('General settings', 'project_path')
    logs_path = os.path.join(info_path, 'logs')
    info_path = os.path.join(logs_path, 'video_info.csv')
    info_df = pd.read_csv(info_path)
    pose_config_path = os.path.join(logs_path, 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    pose_config_df = pd.read_csv(pose_config_path, header=None)
    pose_config_df = list(pose_config_df[0])

    pup_threshold = 0.15
    dam_threshold = 0.2
    roll_windows_values = [1, 2, 5, 8, 0.5]  # values used to calculate rolling average across frames

    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)

    files_found = glob.glob(csv_dir_in + '/*.csv')
    print('Extracting features from ' + str(len(files_found)) + ' files...')

    for currentFile in files_found:
        current_video_file = os.path.basename(currentFile.replace('.csv', ''))
        current_video_settings = info_df.loc[info_df['Video'] == current_video_file]
        try:
            pixels_per_mm = float(current_video_settings['pixels/mm'])
        except TypeError:
            print('Error: Cant find pixels/mm in video settings. Make sure all the videos that are going to be '
                  'analyzed are represented in the project_folder/logs/video_info.csv file')
            break
        fps = float(current_video_settings['fps'])
        
        print('Processing', str(current_video_file), '. Fps:', str(fps), '. mm/ppx:', str(pixels_per_mm))

        # Body parts will be in two categories: dam and pups
        body_part_names = list(pose_config_df)
        dam_body_part_names, pup_body_part_names = [], []
        for bp in body_part_names:
            if 'pup' in bp:
                pup_body_part_names.append(bp)
            else:
                dam_body_part_names.append(bp)

        column_names = []
        for bp in body_part_names:
            column_names.append(bp + '_x')
            column_names.append(bp + '_y')
            column_names.append(bp + '_p')

        csv_df = read_df(currentFile, workflow_type)
        csv_df.columns = column_names

        # csv_df = csv_df.fillna(0)
        csv_df = csv_df.drop(csv_df.index[[0]])
        csv_df = csv_df.apply(pd.to_numeric)
        csv_df = csv_df.reset_index(drop=True)

        print('Calculating dam points and areas')
        
        # Collapse arm and side dam points 
        csv_df['arm_x'] = np.where(csv_df['left_armpit_p'] > dam_threshold, csv_df['left_armpit_x'],
                                   csv_df['right_armpit_x'])
        csv_df['arm_y'] = np.where(csv_df['left_armpit_p'] > dam_threshold, csv_df['left_armpit_y'],
                                   csv_df['right_armpit_y'])
        csv_df['arm_p'] = np.where(csv_df['left_armpit_p'] > dam_threshold, csv_df['left_armpit_p'],
                                   csv_df['right_armpit_p'])

        csv_df['side_x'] = np.where(csv_df['left_ventrum_side_p'] > dam_threshold, csv_df['left_ventrum_side_x'],
                                    csv_df['right_ventrum_side_x'])
        csv_df['side_y'] = np.where(csv_df['left_ventrum_side_p'] > dam_threshold, csv_df['left_ventrum_side_y'],
                                    csv_df['right_ventrum_side_y'])
        csv_df['side_p'] = np.where(csv_df['left_ventrum_side_p'] > dam_threshold, csv_df['left_ventrum_side_p'],
                                    csv_df['right_ventrum_side_p'])

        # Calculate dam centroids and convex hulls
        csv_df['dam_centroid_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                    [row[str(column) + '_x'] for column in dam_body_part_names],
                                                    [row[str(column) + '_p'] for column in dam_body_part_names],
                                                    threshold=dam_threshold), axis=1)
        csv_df['dam_centroid_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                    [row[str(column) + '_y'] for column in dam_body_part_names],
                                                    [row[str(column) + '_p'] for column in dam_body_part_names],
                                                    threshold=dam_threshold), axis=1)

        csv_df['dam_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                                                                [row[p + '_x'] for p in dam_body_part_names],
                                                                [row[p + '_y'] for p in dam_body_part_names],
                                                                [row[p + '_p'] for p in dam_body_part_names]), axis=1)

        centroid_exclusion_parts = ['tail_base', 'left_palm', 'right_palm', 'left_ankle', 'right_ankle']
        dam_core_parts = [p for p in dam_body_part_names if p not in centroid_exclusion_parts]
        csv_df['dam_core_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                                                                [row[p + '_x'] for p in dam_core_parts],
                                                                [row[p + '_y'] for p in dam_core_parts],
                                                                [row[p + '_p'] for p in dam_core_parts]), axis=1)

        dam_head_parts = ['dam_nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'top_head_dam']
        csv_df['head_centroid_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                    [row[str(column) + '_x'] for column in dam_head_parts],
                                                    [row[str(column) + '_p'] for column in dam_head_parts],
                                                    threshold=dam_threshold), axis=1)
        csv_df['head_centroid_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                    [row[str(column) + '_y'] for column in dam_head_parts],
                                                    [row[str(column) + '_p'] for column in dam_head_parts],
                                                    threshold=dam_threshold), axis=1)

        csv_df['head_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                                                                [row[p + '_x'] for p in dam_head_parts],
                                                                [row[p + '_y'] for p in dam_head_parts],
                                                                [row[p + '_p'] for p in dam_head_parts]), axis=1)

        dam_exclusion_parts = ['left_ankle', 'right_ankle', 'tail_base', 'left_palm', 'right_palm', 'btwn_arms',
                               'btwn_legs', 'center_ventrum']
        dam_center_parts = [p for p in dam_body_part_names if p not in dam_exclusion_parts]
        csv_df['dam_center_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                                                                [row[p + '_x'] for p in dam_center_parts],
                                                                [row[p + '_y'] for p in dam_center_parts],
                                                                [row[p + '_p'] for p in dam_center_parts]), axis=1)

        # Calculate the center of all pup points
        print('Calculating pup points')
        
        pup_columns_x, pup_columns_y, pup_columns_p = [], [], []
        for bp in pup_body_part_names:
            pup_columns_x.append(str(bp) + '_x')
            pup_columns_y.append(str(bp) + '_y')
            pup_columns_p.append(str(bp) + '_p')
        
        csv_df['pups_centroid_x'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                                            [row[p] for p in pup_columns_x],
                                                                            [row[p] for p in pup_columns_p],
                                                                            threshold=pup_threshold), axis=1)
        csv_df['pups_centroid_y'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                                            [row[p] for p in pup_columns_y],
                                                                            [row[p] for p in pup_columns_p],
                                                                            threshold=pup_threshold), axis=1)

        print('Calculating pup convex hull and high probably body part counts')
        csv_df['pups_convex_hull'] = csv_df.apply(lambda row: calculate_convex_hull(
                                                                            [row[p] for p in pup_columns_x],
                                                                            [row[p] for p in pup_columns_y],
                                                                            [row[p] for p in pup_columns_p]), axis=1)

        # csv_df['pup_avg_p'] = csv_df.apply(lambda row: calculate_weighted_avg([row[p] for p in pup_columns_p],
        #                                                                       threshold=pup_threshold), axis=1)
        csv_df['pup_avg_p'] = csv_df[[p for p in pup_columns_p]].mean(axis=1)

        csv_df['high_p_pup_bp'] = csv_df.apply(lambda row: count_high_p([row[p] for p in pup_columns_p]), axis=1)
        
        csv_df['high_p_dam_bp'] = csv_df.apply(lambda row: count_high_p([row[p + '_p'] for p in dam_body_part_names]
                                                                        ), axis=1)

        # Calculate movements
        print('Calculating movements')
        movement_columns = dam_body_part_names

        # Create a shifted dataframe and combine
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = [i + '_shifted' for i in csv_df.columns.values.tolist()]
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined = csv_df_combined.reset_index(drop=True)
        
        for bp in movement_columns:
            column_name = 'movement_' + bp
            x1, y1 = (bp + '_x', bp + '_y')
            x2, y2 = (bp + '_x_shifted', bp + '_y_shifted')
            csv_df[column_name] = (np.sqrt((csv_df_combined[x1] - csv_df_combined[x2]) ** 2 +
                                           (csv_df_combined[y1] - csv_df_combined[y2]) ** 2)) / pixels_per_mm

        csv_df['back_avg_movement'] = np.ma.average([csv_df['movement_back_2'],
                                                     csv_df['movement_back_4'],
                                                     csv_df['movement_back_8'],
                                                     csv_df['movement_back_10']], axis=0)

        csv_df['head_avg_movement'] = np.ma.average([csv_df['movement_dam_nose'],
                                                     csv_df['movement_right_eye'],
                                                     csv_df['movement_left_ear'],
                                                     csv_df['movement_right_ear']], axis=0)

        csv_df['head_max_movement'] = np.ma.max([csv_df['movement_dam_nose'],
                                                     csv_df['movement_right_eye'],
                                                     csv_df['movement_left_ear'],
                                                     csv_df['movement_right_ear']], axis=0)

        # Moving average of movement
        print('Calculating moving average of movements')
        
        roll_windows = []
        for j in range(len(roll_windows_values)):
            roll_windows.append(int(fps / roll_windows_values[j]))
            
        csv_df['head_avg_movement_mavg_30'] = csv_df['head_avg_movement'].rolling(roll_windows[0], min_periods=1).mean()
        csv_df['head_avg_movement_mavg_6'] = csv_df['head_avg_movement'].rolling(roll_windows[2], min_periods=1).mean()
        csv_df['head_avg_movement_mavg_60'] = csv_df['head_avg_movement'].rolling(roll_windows[4], min_periods=1).mean()

        csv_df['head_max_movement_mavg_30'] = csv_df['head_max_movement'].rolling(roll_windows[0], min_periods=1).mean()
        csv_df['head_max_movement_mavg_6'] = csv_df['head_max_movement'].rolling(roll_windows[2], min_periods=1).mean()
        csv_df['head_max_movement_mavg_60'] = csv_df['head_max_movement'].rolling(roll_windows[4], min_periods=1).mean()

        csv_df['back_avg_movement_mavg_30'] = csv_df['back_avg_movement'].rolling(roll_windows[0], min_periods=1).mean()
        csv_df['back_avg_movement_mavg_6'] = csv_df['back_avg_movement'].rolling(roll_windows[2], min_periods=1).mean()
        csv_df['back_avg_movement_mavg_60'] = csv_df['back_avg_movement'].rolling(roll_windows[4], min_periods=1).mean()

        csv_df['head_back_rel_mov_30'] = csv_df['head_avg_movement_mavg_30'] / (csv_df['head_avg_movement_mavg_30'] +
                                                                                csv_df['back_avg_movement_mavg_30'])
        csv_df['head_back_rel_mov_6'] = csv_df['head_avg_movement_mavg_6'] / (csv_df['head_avg_movement_mavg_6'] +
                                                                              csv_df['back_avg_movement_mavg_6'])
        csv_df['head_back_rel_mov_60'] = csv_df['head_avg_movement_mavg_2'] / (csv_df['head_avg_movement_mavg_60'] +
                                                                               csv_df['back_avg_movement_mavg_60'])

        csv_df['pups_convex_hull_mavg_30'] = csv_df['pups_convex_hull'].rolling(roll_windows[0], min_periods=1).mean()
        csv_df['pups_convex_hull_mavg_6'] = csv_df['pups_convex_hull'].rolling(roll_windows[2], min_periods=1).mean()
        csv_df['pups_convex_hull_mavg_60'] = csv_df['pups_convex_hull'].rolling(roll_windows[4], min_periods=1).mean()

        # Distance calculations
        print('Calculating distances')
        csv_df['dam_pup_distance'] = np.sqrt((csv_df['dam_centroid_x'] - csv_df['pups_centroid_x'])**2 +
                                             (csv_df['dam_centroid_y'] - csv_df['pups_centroid_y'])**2)
        csv_df['head_pup_distance'] = np.sqrt((csv_df['head_centroid_x'] - csv_df['pups_centroid_x'])**2 +
                                              (csv_df['head_centroid_y'] - csv_df['pups_centroid_y'])**2)

        csv_df['back_length'] = np.sqrt((csv_df['back_2_x'] - csv_df['back_10_x'])**2
                                        + (csv_df['back_2_y'] - csv_df['back_10_y'])**2)

        csv_df['nose_back10_length'] = np.sqrt((csv_df['dam_nose_x'] - csv_df['back_10_x'])**2
                                               + (csv_df['dam_nose_y'] - csv_df['back_10_y'])**2)

        csv_df['back1_back10_length'] = np.sqrt((csv_df['back_1_center_x'] - csv_df['back_10_x'])**2
                                                + (csv_df['back_1_center_y'] - csv_df['back_10_y'])**2)

        csv_df['nose_back2_length'] = np.sqrt((csv_df['dam_nose_x'] - csv_df['back_2_x'])**2
                                              + (csv_df['dam_nose_y'] - csv_df['back_2_y'])**2)

        csv_df['left_wrist_nose_length'] = np.sqrt((csv_df['left_wrist_x'] - csv_df['dam_nose_x'])**2 +
                                                   (csv_df['left_wrist_y'] - csv_df['dam_nose_y'])**2)
        csv_df['right_wrist_nose_length'] = np.sqrt((csv_df['right_wrist_x'] - csv_df['dam_nose_x'])**2 +
                                                    (csv_df['right_wrist_y'] - csv_df['dam_nose_y'])**2)
        csv_df['wrist_nose_length'] = csv_df.apply(lambda row: calculate_weighted_avg(
                                                [row['left_wrist_nose_length'], row['right_wrist_nose_length']],
                                                [row['left_wrist_p'], row['right_wrist_p']],
                                                threshold=dam_threshold), axis=1)
        csv_df.drop(inplace=True, columns=['left_wrist_nose_length', 'right_wrist_nose_length'])

        csv_df['avg_dam_bp_p'] = np.ma.average([csv_df['dam_nose_p'],
                                                csv_df['left_eye_p'],
                                                csv_df['right_eye_p'],
                                                csv_df['left_ear_p'],
                                                csv_df['right_ear_p'],
                                                csv_df['left_shoulder_p'],
                                                csv_df['right_shoulder_p'],
                                                csv_df['arm_p'],
                                                csv_df['side_p']], axis=0)

        csv_df['sum_probabilities'] = csv_df[[p + '_p' for p in body_part_names]].sum(axis=1)

        # Save DF
        print('Exporting df')
        csv_df = csv_df.reset_index(drop=True)
        csv_df = csv_df.fillna(0)
        output_file_name = os.path.basename(currentFile).replace('.' + workflow_type, '')
        save_path = os.path.join(csv_dir_out, output_file_name) + '.csv'
        print('Save path:', save_path)
        save_df(csv_df, workflow_type, save_path)
        print('Feature extraction complete for', str(current_video_file))

    print('All feature extraction complete.')
