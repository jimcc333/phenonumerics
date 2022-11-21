# Phenonumerics property
# Written by Cem Bagdatlioglu
#
# 2021-08-21
#
# FOR INTERNAL USE ONLY.
#

import sys
import os


def get_data_folder():
    args = sys.argv
    if len(args) != 2:
        raise ValueError('Need exactly one input for folder file path.')
    return args[1]


def video_folder_checker(video_folder_path):
    print('Checking', video_folder_path, ':')

    if not os.path.isdir(video_folder_path):
        raise ValueError(video_folder_path, 'is not a directory.')

    files = os.listdir(video_folder_path)
    print('  Found', len(files), 'files.')

    # Find csv files
    csv_files = []
    for file in files:
        if file[-3:] == 'csv':
            csv_files.append(file)
    if len(csv_files) == 0:
        raise ValueError('csv file not found')
    if len(csv_files) > 1:
        raise ValueError('more than one csv file found')

    csv_file = open(video_folder_path + '\\' + csv_files[0], 'r')
    line_number = 0
    csv_image_files = []
    for line in csv_file.readlines():
        line_number = line_number + 1
        if line_number >= 5:
            csv_image_files.append(line.strip().split(',')[0].split('\\')[-1])

    # Find image names in folder
    folder_image_files = []
    for file in files:
        if file[-3:] == 'png':
            folder_image_files.append(file)

    # Check if numbers match
    if len(csv_image_files) != len(folder_image_files):
        raise ValueError(len(csv_image_files), ' image files vs ', len(folder_image_files), ' files in csv')

    # Check if names match
    csv_image_files.sort()
    folder_image_files.sort()
    for i in range(len(csv_image_files)):
        # print('   checking', csv_image_files[i], 'and', folder_image_files[i])
        if csv_image_files[i] != folder_image_files[i]:
            raise ValueError('File name mismatch, in CSV:', csv_image_files[i], 'in folder:', folder_image_files[i])

    # Find number of columns
    csv_file.seek(0)

    return len(csv_file.readline().split(',')) - 1


if __name__ == '__main__':
    print('\n\n\n\n\n\n\n\n\n\n\n\n---------------------------------------\n\n')
    print('Welcome to Awesome PhenoVid Checker 3000!\n Let''s go!!\n')
    data_folder = get_data_folder()
    video_folders = os.listdir(data_folder)
    #add further folder vs file check
    print('There are ', len(video_folders), ' video folders found.')

    # iterate through all the video folders
    columns = {}
    for video_folder_path in video_folders:
        if video_folder_path[-8:] == '_labeled':
            print('Skipping', video_folder_path)
        else:
            print(video_folder_path)
            columns[video_folder_path] = video_folder_checker(data_folder + '\\' + video_folder_path)

    # Report mismatched columns
    iterator = iter(set(columns.values()))
    column_count = next(iterator, None)
    if len(set(columns.values())) == 1:
        print('\nAll folders have', column_count, 'columns.')
    else:
        print('Column mismatch.')
        for k, v in columns.items():
            print(v, k)

    print('\n..Goodbye..')


