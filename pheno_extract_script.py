from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from configparser import ConfigParser
import glob
from simba.rw_dfs import *


def extract_features_userdef(inifile):
    print('Phenosimba says hello, yo! This is version 4.6')
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

    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)

    def count_values_in_range(series, values_in_range_min, values_in_range_max):
        return series.between(left=values_in_range_min, right=values_in_range_max).sum()

    roll_windows = []
    roll_windows_values = [2, 5, 6, 7.5, 15]
    loopy = 0

    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Extracting features from ' + str(len(filesFound)) + ' files...')

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    for i in filesFound:
        currentFile = i
        currVidName = os.path.basename(currentFile.replace('.csv', ''))
        currVideoSettings = vidinfDf.loc[vidinfDf['Video'] == currVidName]
        try:
            currPixPerMM = float(currVideoSettings['pixels/mm'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        fps = float(currVideoSettings['fps'])
        print('Processing ' + '"' + str(currVidName) + '".' + ' Fps: ' + str(fps) + ". mm/ppx: " + str(currPixPerMM))
        for j in range(len(roll_windows_values)):
            roll_windows.append(int(fps / roll_windows_values[j]))
        loopy += 1

        # exclude pup body parts from bodypartNames
        bodypartNames = list(poseConfigDf)
        final_body_part_names = []
        for part in bodypartNames:
            if 'pup' not in part:
                final_body_part_names.append(part)

        columnHeaders = []
        columnHeadersShifted = []
        p_cols = []
        for bodypart in bodypartNames:
            colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
            colHead4, colHead5, colHead6 = (bodypart + '_x_shifted', bodypart + '_y_shifted', bodypart + '_p_shifted')
            columnHeaders.extend((colHead1, colHead2, colHead3))
            columnHeadersShifted.extend((colHead4, colHead5, colHead6))
            p_cols.append(colHead3)
        csv_df = read_df(currentFile, wfileType)
        csv_df.columns = columnHeaders

        csv_df = csv_df.fillna(0)
        csv_df = csv_df.drop(csv_df.index[[0]])
        csv_df = csv_df.apply(pd.to_numeric)
        csv_df = csv_df.reset_index(drop=True)


        ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = columnHeadersShifted
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined = csv_df_combined.reset_index(drop=True)


        ########### MOVEMENTS OF ALL BODY PARTS ###########################################
        movementColNames = []
        for selectBp in bodypartNames:
            colName = 'movement_' + selectBp
            movementColNames.append(colName)
            selectBpX_1, selectBpY_1 = (selectBp + '_x', selectBp + '_y')
            selectBpX_2, selectBpY_2 = (selectBp + '_x_shifted', selectBp + '_y_shifted')
            csv_df[colName] = (np.sqrt((csv_df_combined[selectBpX_1] - csv_df_combined[selectBpX_2]) ** 2 + (csv_df_combined[selectBpY_1] - csv_df_combined[selectBpY_2]) ** 2)) / currPixPerMM
        movementDf = csv_df.filter(movementColNames, axis=1)
        descriptiveColNames = ['collapsed_sum_of_all_movements', 'collapsed_mean_of_all_movements', 'collapsed_median_of_all_movements', 'collapsed_min_of_all_movements', 'collapsed_max_of_all_movements']
        csv_df['collapsed_sum_of_all_movements'] = movementDf[movementColNames].sum(axis=1)
        csv_df['collapsed_mean_of_all_movements'] = movementDf[movementColNames].mean(axis=1)
        csv_df['collapsed_median_of_all_movements'] = movementDf[movementColNames].median(axis=1)
        csv_df['collapsed_min_of_all_movements'] = movementDf[movementColNames].min(axis=1)
        csv_df['collapsed_max_of_all_movements'] = movementDf[movementColNames].max(axis=1)


        ########### CALC THE NUMBER OF LOW PROBABILITY DETECTIONS & TOTAL PROBABILITY VALUE FOR ROW###########################################
        print('Calculating pose probability scores...')
        probabilityDf = csv_df.filter(p_cols, axis=1)
        csv_df['Sum_probabilities'] = probabilityDf.sum()
        csv_df['Mean_probabilities'] = probabilityDf.mean()
        values_in_range_min, values_in_range_max = 0.0, 0.1
        csv_df["Low_prob_detections_0.1"] = probabilityDf.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_min, values_in_range_max = 0.000000000, 0.5
        csv_df["Low_prob_detections_0.5"] = probabilityDf.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_min, values_in_range_max = 0.000000000, 0.75
        csv_df["Low_prob_detections_0.75"] = probabilityDf.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)


        ########### CALC CENTROIDS ################################
        print('Calculating dam centroid')
        csv_df['dam_centroid_x'] = np.ma.average([csv_df['dam_nose_x'],
                                             csv_df['left_eye_x'],
                                             csv_df['right_eye_x'],
                                             csv_df['left_ear_x'],
                                             csv_df['right_ear_x'],
                                             csv_df['left_shoulder_x'],
                                             csv_df['right_shoulder_x'],
                                             csv_df['arm_x'],
                                             csv_df['side_x'],
                                             csv_df['back_2_x'],
                                             csv_df['back_4_x'],
                                             csv_df['back_8_x'],
                                             csv_df['back_10_x']],
                                             weights=[csv_df['dam_nose_p'],
                                             csv_df['left_eye_p'],
                                             csv_df['right_eye_p'],
                                             csv_df['left_ear_p'],
                                             csv_df['right_ear_p'],
                                             csv_df['left_shoulder_p'],
                                             csv_df['right_shoulder_p'],
                                             csv_df['arm_p'],
                                             csv_df['side_p'],
                                             csv_df['back_2_p'],
                                             csv_df['back_4_p'],
                                             csv_df['back_8_p'],
                                             csv_df['back_10_p']],
                                              axis=0)
        csv_df['dam_centroid_y'] = np.ma.average([csv_df['dam_nose_y'],
                                             csv_df['left_eye_y'],
                                             csv_df['right_eye_y'],
                                             csv_df['left_ear_y'],
                                             csv_df['right_ear_y'],
                                             csv_df['left_shoulder_y'],
                                             csv_df['right_shoulder_y'],
                                             csv_df['arm_y'],
                                             csv_df['side_y'],
                                             csv_df['back_2_y'],
                                             csv_df['back_4_y'],
                                             csv_df['back_8_y'],
                                             csv_df['back_10_y']],
                                             weights=[csv_df['dam_nose_p'],
                                             csv_df['left_eye_p'],
                                             csv_df['right_eye_p'],
                                             csv_df['left_ear_p'],
                                             csv_df['right_ear_p'],
                                             csv_df['left_shoulder_p'],
                                             csv_df['right_shoulder_p'],
                                             csv_df['arm_p'],
                                             csv_df['side_p'],
                                             csv_df['back_2_p'],
                                             csv_df['back_4_p'],
                                             csv_df['back_8_p'],
                                             csv_df['back_10_p']],
                                              axis=0)

        print('Calculating pups centroid')
        csv_df['pups_centroid_x'] = np.ma.average([ csv_df['pup1_x'],
                                                 csv_df['pup2_x'],
                                                 csv_df['pup3_x'],
                                                 csv_df['pup4_x'],
                                                 csv_df['pup5_x'],
                                                 csv_df['pup6_x'],
                                                 csv_df['pup7_x'],
                                                 csv_df['pup8_x']],
                                             weights=[csv_df['pup1_p'],
                                                     csv_df['pup2_p'],
                                                     csv_df['pup3_p'],
                                                     csv_df['pup4_p'],
                                                     csv_df['pup5_p'],
                                                     csv_df['pup6_p'],
                                                     csv_df['pup7_p'],
                                                     csv_df['pup8_p']],
                                              axis=0)
        csv_df['pups_centroid_y'] = np.ma.average([ csv_df['pup1_y'],
                                                 csv_df['pup2_y'],
                                                 csv_df['pup3_y'],
                                                 csv_df['pup4_y'],
                                                 csv_df['pup5_y'],
                                                 csv_df['pup6_y'],
                                                 csv_df['pup7_y'],
                                                 csv_df['pup8_y']],
                                             weights=[csv_df['pup1_p'],
                                                     csv_df['pup2_p'],
                                                     csv_df['pup3_p'],
                                                     csv_df['pup4_p'],
                                                     csv_df['pup5_p'],
                                                     csv_df['pup6_p'],
                                                     csv_df['pup7_p'],
                                                     csv_df['pup8_p']],
                                              axis=0)

        # Calculate pups convex hull and avg p
        print('Calculating pups convext hull')
        pup_threshold = 0.1
        dam_threshold = 0.5
        pups_convex_hull = []
        pup_avg_p = []
        high_p_pups = []
        high_p_dam_bp = []
        for id, row in csv_df.iterrows():
            points = []
            ps = []
            bps = 0
            for pup_i in range(8):
                if row['pup' + str(pup_i+1) + '_p'] > pup_threshold:
                    points.append([row['pup' + str(pup_i+1) + '_x'], row['pup' + str(pup_i+1) + '_y']])
                    ps.append(row['pup' + str(pup_i+1) + '_p'])
            pup_avg_p.append(np.average(ps))
            high_p_pups.append(len(points))
            if len(points) < 3:
                pups_convex_hull.append(0)
            else:
                pups_convex_hull.append(ConvexHull(points).volume)
            for bodypart in bodypartNames:
                if row[bodypart + '_p'] > dam_threshold:
                    bps += 1
            high_p_dam_bp.append(bps)

        csv_df['pups_convex_hull'] = pups_convex_hull
        csv_df['pup_avg_p'] = pup_avg_p
        csv_df['high_p_pups'] = high_p_pups
        csv_df['high_p_dam_bp'] = high_p_dam_bp

        ########### CALC AVG MOVEMENTS ################################
        print('Calculating avg movements and distances')

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

        csv_df['dam_pup_distance'] = np.sqrt((csv_df['dam_centroid_x'] - csv_df['pups_centroid_x'])**2
                                             + (csv_df['dam_centroid_y'] - csv_df['pups_centroid_y'])**2)

        csv_df['back_length'] = np.sqrt((csv_df['back_2_x'] - csv_df['back_10_x'])**2
                                             + (csv_df['back_2_y'] - csv_df['back_10_y'])**2)

        csv_df['nose_back10_length'] = np.sqrt((csv_df['dam_nose_x'] - csv_df['back_10_x'])**2
                                             + (csv_df['dam_nose_y'] - csv_df['back_10_y'])**2)

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


        ########### SAVE DF ###########################################
        print('Exporting df')
        #csv_df = csv_df.loc[:, ~csv_df.T.duplicated(keep='first')]
        csv_df = csv_df.reset_index(drop=True)
        csv_df = csv_df.fillna(0)
        #csv_df = csv_df.drop(columns=['index'])
        fileOutName = os.path.basename(currentFile).replace('.' + wfileType, '')
        savePath = os.path.join(csv_dir_out, fileOutName)
        print('Save path:', savePath)
        save_df(csv_df, wfileType, savePath)
        print('Feature extraction complete for ' + '"' + str(currVidName) + '".')

    print('All feature extraction complete.')