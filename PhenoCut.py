# Phenonumerics property
# Written by Cem Bagdatlioglu
#
# 2021-08-28
#
# FOR INTERNAL USE ONLY.
#

import sys
import os
import shutil
import pandas as pd


def confirm_choice(message, default='y'):
    choices = 'Y/N'
    choice = input("%s (%s) " % (message, choices))
    values = ('y', 'yes', '')
    return choice.strip().lower() in values


def convert_separator_type(video_folders_path):
    video_folders = os.listdir(video_folders_path)
    #TODO: add further folder vs file check
    print('There are', len(video_folders), 'video folders found in', video_folders_path)

    if not confirm_choice('Are you sure you want to convert slashes in the CSVs?'):
        return

    # iterate through all the video folders
    for video_folder_path in video_folders:
        full_video_folder_path = video_folders_path + os.sep + video_folder_path
        if not os.path.isdir(full_video_folder_path):
            print('  Skipping', full_video_folder_path)
            continue
        print('  Converting paths in', full_video_folder_path)

        files = os.listdir(full_video_folder_path)
        print('  Found', len(files), 'files.')

        # Find csv files
        csv_files = []
        for file in files:
            if file[-3:] == 'csv':
                csv_files.append(file)
        if len(csv_files) == 0:
            print('  No CSV file found, skipping folder.')
            continue

        csv_file_path = full_video_folder_path + os.sep + csv_files[0]
        print('   Opening', csv_file_path)
        csv_file = open(csv_file_path, 'r')
        new_file_content = ''
        for line in csv_file.readlines():
            new_line = line.strip().replace('\\', os.sep).replace('/', os.sep)
            new_file_content += new_line + '\n'

        new_file = open(csv_file_path, 'w')
        new_file.write(new_file_content)
        new_file.close()


def get_csv_path(directory_path):
    files = os.listdir(directory_path)

    # Find csv files
    csv_files = []
    for file in files:
        if file[-3:] == 'csv':
            csv_files.append(file)
    if len(csv_files) == 0:
        raise ValueError('csv file not found at', directory_path)
    if len(csv_files) > 1:
        raise ValueError('more than one csv file found')

    return directory_path + os.sep + csv_files[0]


def video_folder_checker(video_folder_path):
    if video_folder_path.split(os.sep)[-1][0] == '.':
        print('  Skipping', video_folder_path)
        return -1
    if not os.path.isdir(video_folder_path):
        raise ValueError(video_folder_path + ' is not a directory.')

    files = os.listdir(video_folder_path)
    print('    Found', len(files), 'files.')

    csv_file = open(get_csv_path(video_folder_path), 'r')
    line_number = 0
    csv_image_files = []
    for line in csv_file.readlines():
        line_number = line_number + 1
        if line_number >= 5:
            csv_image_files.append(line.strip().split(',')[0].split(os.sep)[-1])

    # Find image names in folder
    folder_image_files = []
    for file in files:
        if file[-3:] == 'png':
            folder_image_files.append(file)

    # Check if numbers match
    if len(csv_image_files) != len(folder_image_files):
        raise ValueError(str(len(csv_image_files)) + ' image files vs ' + str(len(folder_image_files))
                         + ' files in csv at ' + video_folder_path)

    # Check if names match
    csv_image_files.sort()
    folder_image_files.sort()
    for i in range(len(csv_image_files)):
        # print('   checking', csv_image_files[i], 'and', folder_image_files[i])
        if csv_image_files[i] != folder_image_files[i]:
            raise ValueError('File name mismatch! Name in CSV: ' + csv_image_files[i]
                             + ' Name in folder: ' + folder_image_files[i])
    print('    All images match the CSV')

    # Help find number of columns
    csv_file.seek(0)

    return len(csv_file.readline().split(',')) - 1


def video_folders_checker(video_folders_path):

    video_folders = os.listdir(video_folders_path)
    #TODO: add further folder vs file check
    print('There are ', len(video_folders), ' video folders found in ', video_folders_path)

    # iterate through all the video folders
    columns = {}
    for video_folder_path in video_folders:
        if video_folder_path[-8:] == '_labeled':
            print('  Skipping', video_folder_path)
        else:
            full_video_folder_path = video_folders_path + os.sep + video_folder_path
            print('  Checking', full_video_folder_path)
            new_columns = video_folder_checker(full_video_folder_path)
            if new_columns >= 0:
                columns[video_folder_path] = video_folder_checker(full_video_folder_path)

    # Report mismatched columns
    iterator = iter(set(columns.values()))
    column_count = next(iterator, None)
    if len(set(columns.values())) == 1:
        print('\nAll folders have', column_count, 'columns.')
        return column_count
    else:
        print('Column mismatch.')
        for k, v in columns.items():
            print(v, k)
        return 0


def read_config_file(config_path):
    print('Reading', config_path)
    config_file = open(config_path, 'r')
    line_number = 0
    read_individuals = False
    read_unique_parts = False
    read_individual_parts = False
    scorer_name = 'not-found-in-config'

    individuals = []
    unique_parts = []
    individual_parts = []

    for line in config_file.readlines():
        current_line = line.strip()
        line_number += 1

        # Skip comments and empty lines
        if len(current_line) == 0:
            continue
        if current_line[0] == '#':
            continue

        # Get single-line info
        if current_line[:7].lower() == 'scorer:':
            scorer_name = current_line[8:]
            print('Scorer:', scorer_name, '\n')

        # Below is the body parts logic
        if current_line[0] != '-':
            read_individuals = False
            read_unique_parts = False
            read_individual_parts = False

        if read_individuals:
            individuals.append(current_line[2:])

        if read_unique_parts:
            unique_parts.append(current_line[2:])

        if read_individual_parts:
            individual_parts.append(current_line[2:])

        if current_line.lower() == 'individuals:':
            read_individuals = True
            read_unique_parts = False
            read_individual_parts = False

        if current_line.lower() == 'uniquebodyparts:':
            read_individuals = False
            read_unique_parts = True
            read_individual_parts = False

        if current_line.lower() == 'multianimalbodyparts:':
            read_individuals = False
            read_unique_parts = False
            read_individual_parts = True

    print(len(individuals), 'individuals:', individuals, '\n')
    print(len(unique_parts), 'unique body parts:', unique_parts, '\n')
    print(len(individual_parts), 'multianimal body parts:', individual_parts, '\n')

    # Build expected top 4 lines
    total_labels = len(individuals) * len(individual_parts) + len(unique_parts)
    total_coordinates = total_labels * 2

    first_line = [scorer_name] * total_coordinates
    first_line.insert(0, 'scorer')

    second_line = []
    for individual in individuals:
        second_line.extend(individual for i in range(len(individual_parts) * 2))
    second_line.extend(['single'] * len(unique_parts) * 2)
    second_line.insert(0, 'individuals')

    third_line = []
    for part in individual_parts:
        third_line.append(part)
        third_line.append(part)
    third_line = [i for j in range(len(individuals)) for i in third_line]  # this copies the line len(individuals) times
    for unique_part in unique_parts:
        third_line.append(unique_part)
        third_line.append(unique_part)
    third_line.insert(0, 'bodyparts')

    fourth_line = ['x', 'y'] * (len(individuals) * len(individual_parts) + len(unique_parts))
    fourth_line.insert(0, 'coords')

    return first_line, second_line, third_line, fourth_line


def config_fix(project_path):
    first_line, second_line, third_line, fourth_line = read_config_file(project_path + os.sep + 'config.yaml')

    # iterate through all video file folders and check against
    video_folders_path = project_path + os.sep + 'labeled-data'

    video_folders = os.listdir(video_folders_path)
    columns = {}
    for video_folder_path in video_folders:
        full_video_folder_path = video_folders_path + os.sep + video_folder_path
        if video_folder_path[-8:] == '_labeled':
            shutil.rmtree(full_video_folder_path)
            print('Deleted', full_video_folder_path)
        else:
            print('Checking', video_folder_path)
            new_columns = video_folder_checker(full_video_folder_path)
            if new_columns == -1:
                continue
            if new_columns == len(first_line) - 1:
                print('    Column counts match')
            csv_path = get_csv_path(project_path + os.sep + 'labeled-data' + os.sep + video_folder_path)
            csv_file = open(csv_path, 'r')
            line_number = 0
            csv_first_line = []
            csv_second_line = []
            csv_third_line = []
            csv_fourth_line = []
            for line in csv_file.readlines():
                line_number += 1
                if line_number == 1:
                    csv_first_line = line.split(',')
                    csv_first_line[-1] = csv_first_line[-1].strip()
                if line_number == 2:
                    csv_second_line = line.split(',')
                    csv_second_line[-1] = csv_second_line[-1].strip()
                if line_number == 3:
                    csv_third_line = line.split(',')
                    csv_third_line[-1] = csv_third_line[-1].strip()
                if line_number == 4:
                    csv_fourth_line = line.split(',')
                    csv_fourth_line[-1] = csv_fourth_line[-1].strip()
            csv_file.close()

            # Delete extra columns
            pd_csv = pd.read_csv(csv_path)
            csv_original_len = len(csv_first_line)
            if csv_original_len == 0:
                print('Could not find csv at', csv_path)
                return

            remove_count = 0
            for i in range(len(first_line)):
                while first_line[i] != csv_first_line[i] \
                        or second_line[i] != csv_second_line[i] \
                        or third_line[i] != csv_third_line[i] \
                        or fourth_line[i] != csv_fourth_line[i]:
                    remove_count += 1
                    pd_csv.drop(pd_csv.columns[i], inplace=True, axis=1)
                    csv_first_line.remove(csv_first_line[i])
                    csv_second_line.remove(csv_second_line[i])
                    csv_third_line.remove(csv_third_line[i])
                    csv_fourth_line.remove(csv_fourth_line[i])
            print('Removed', remove_count, 'columns')
            if len(csv_first_line) != csv_original_len:
                print('Writing new csv:', csv_path)
                # os.rename(csv_path, csv_path[:-4] + '_OLD.csv')
                pd_csv.columns = pd_csv.columns.str.split('.').str[0]
                pd_csv.to_csv(csv_path, index=False, encoding='utf8')

            for i in range(len(first_line)):
                if first_line[i] != csv_first_line[i]:
                    print('Column mismatch on first row! Column', i, csv_path)
                if second_line[i] != csv_second_line[i]:
                    print('Column mismatch on second row! Column', i, csv_path)
                if third_line[i] != csv_third_line[i]:
                    print('Column mismatch on third row! Column', i, csv_path)
                if fourth_line[i] != csv_fourth_line[i]:
                    print('Column mismatch on fourth row! Column', i, csv_path)
            print('    All column names match')



def remove_h5(video_folders_path):
    video_folders = os.listdir(video_folders_path)
    #TODO: add further folder vs file check
    print('There are', len(video_folders), 'video folders found in', video_folders_path)
    # iterate through all the video folders
    h5_file_paths = []
    for video_folder_path in video_folders:
        full_video_folder_path = video_folders_path + os.sep + video_folder_path
        if not os.path.isdir(full_video_folder_path):
            continue

        files = os.listdir(full_video_folder_path)

        # Find h5 files
        for file in files:
            if file[-3:] == '.h5':
                h5_file_paths.append(full_video_folder_path + os.sep + file)

    if len(h5_file_paths) == 0:
        print('Could not find any .h5 files in', video_folders_path)
        return

    warning_message = 'Found ' + str(len(h5_file_paths)) + ' .h5 files. Are you sure you want to delete all of them?'
    if not confirm_choice(warning_message):
        return

    for path in h5_file_paths:
        os.remove(path)
        print('Deleted:', path, '\n')


def show_help():
    print('\n-PhenoCut Help-')
    print('[PhenoCut.py [args] [file path]')
    print('Arguments:')
    print('-c: Read the config file. The [file path] is for the config.yaml file.')
    print('-cf: Config fix. Uses config file to remove columns from the video folders. Pass the project directory.')
    print('-h5: Remove all .h5 files from the [file path] of the video frame folders directory (labeled-data).')
    print('-sep: Converts the file path separators to the native system type in the provided video folders directory.')
    print('-v: The [file path] is for the directory containing the video frame folders (labeled-data).')
    print('-h: Show help')


def main(args):
    rows, columns = os.popen('stty size', 'r').read().split()
    print('\n\n\n\n')
    print('-'*int(columns))
    print('Welcome to PhenoCut')
    print('-'*int(columns))

    if len(args) == 1:
        show_help()
        return

    input_arg = args[1]

    if input_arg == '-h':
        show_help()
        return

    if input_arg == '-c':
        try:
            input_path = args[2]
        except IndexError:
            print('Error! Specify the full path for the config.yaml file.')
            return
        read_config_file(input_path)
        return

    if input_arg == '-cf':
        try:
            input_path = args[2]
        except IndexError:
            print('Error! Specify the full path for the project directory.')
            return
        config_fix(input_path)
        return

    if input_arg == '-h5':
        try:
            input_path = args[2]
        except IndexError:
            print('Error! Specify the full path for the video frame folders directory (labeled-data).')
            return
        remove_h5(input_path)
        return

    if input_arg == '-sep':
        try:
            input_path = args[2]
        except IndexError:
            print('Error! Specify the full path for the directory where the folders containing video frames and CSVs '
                  'as the second argument.')
            return
        convert_separator_type(input_path)
        return

    if '-v' in input_arg:
        try:
            input_path = args[2]
        except IndexError:
            print('Error! Specify the full path for the directory where the folders containing video frames and CSVs '
                  'as the second argument.')
            return
        video_folders_checker(input_path)
        return


def run_phenocut():
    sys.exit(main(sys.argv))


if __name__ == '__main__':
    run_phenocut()

