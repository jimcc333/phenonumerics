# by Cem B
import pickle
import sys

import numpy as np
import pandas as pd
import os
import ruamel.yaml

def show_help():
    print('PhenoPickkllee!')
    print('Must specificy at least the input and output full paths:')
    print(' -input:[pickled file full path]')
    print(' -output:[unpickled file full directory path]')
    print(' -max_tracks:#')
    print(' -empty_cell_val:[]')
    print()
    print('Example: python PhenoPickle.py -input:/full/path/to/input.pickle -output:full/path/to/output.csv -max_tracks:11 -empty_cell_val:null')
    return


# Press the green button in the gutter to run the script.
def main(args, max_tracks=12, empty_cell_val='NA', file_path=None, output_path=None):
    print(args)
    if len(args) == 1:
        show_help()
        return

    if args[1] == '-h':
        show_help()
        return

    for arg in args:
        if arg[0:6] == '-input':
            file_path = arg[7:]

    for arg in args:
        if arg[0:7] == '-output':
            output_path = arg[8:]
            if output_path[-4:] != '.csv':
                print(output_path)
                print('Output does not end with .csv!', output_path[-4:])
                return

    for arg in args:
        if arg[0:12] == '-max_tracks':
            max_tracks = arg[13:]
            print('Max tracks:', max_tracks)

    for arg in args:
        if arg[0:17] == '-empty_cell_val':
            empty_cell_val = arg[18:]
            print('Empty cell val:', empty_cell_val)

    if file_path is None:
        print('No file path')
        return
    if output_path is None:
        print('No output path')
        return

    print('Looking at:', output_path)
    print(output_path)

    df = pd.read_pickle(file_path)

    print('Read pickle file with', len(df), 'tracklets')

    body_parts = ['nose', 'eyes', 'ears', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6']

    columns = ['frame']
    for pup in range(max_tracks):
        for part in body_parts:
            columns.append('pup' + str(pup+1) + '_' + part + '_x')
            columns.append('pup' + str(pup+1) + '_' + part + '_y')
            columns.append('pup' + str(pup+1) + '_' + part + '_likelihood')

    output = []

    counter = 0
    for k, v in df.items():
        counter += 1
        row = [k]

        for arr in v:
            for i in range(len(arr)):
                for j in range(3):
                    row.append(arr[i][j])

        empty_needed = len(columns) - len(row)
        for i in range(empty_needed):
            row.append(np.NAN)
        output.append(row)

    output_file = pd.DataFrame(output, columns=columns)
    output_file.fillna(empty_cell_val, inplace=True)
    print(output_file)
    print('Writing output to:', output_path)

    output_file.to_csv(output_path)
    print('Done unpickling!')
    return


def run_phenopickle():
    sys.exit(main(sys.argv))


if __name__ == '__main__':
    run_phenopickle()
