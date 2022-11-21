from __future__ import division
import os
import pandas as pd
import numpy as np
from configparser import ConfigParser
import glob
from simba.rw_dfs import *


def extract_features_userdef(inifile):
    print('Phenosimba says hello, yo! This is version 2.1')
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


        ########### SAVE DF ###########################################
        #csv_df = csv_df.loc[:, ~csv_df.T.duplicated(keep='first')]
        csv_df = csv_df.reset_index(drop=True)
        csv_df = csv_df.fillna(0)
        #csv_df = csv_df.drop(columns=['index'])
        fileOutName = os.path.basename(currentFile).replace('.' + wfileType, '')
        savePath = os.path.join(csv_dir_out, fileOutName)
        save_df(csv_df, wfileType, savePath)
        print('Feature extraction complete for ' + '"' + str(currVidName) + '".')

    print('All feature extraction complete.')