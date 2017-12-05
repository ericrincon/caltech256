"""
Short script to move files to one folder
"""

import os
import argparse
import shutil

from util import get_all_files, parse_file_path, \
    is_image, get_subdirs

from sklearn.model_selection import train_test_split


def move_files(file_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, file_path in enumerate(file_paths):
        file_name = parse_file_path(file_path)

        # move the file to the output dir if it's a JPEG
        if '_' in file_name and not file_name == '256_ObjectCategories':
            output_file_path = "{}/{}".format(output_dir, file_name)

            if os.path.isfile(file_path):
                shutil.move(file_path, output_file_path)
                print("{}: Moved file {}!".format(i, file_name))


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--input-dir')
    argument_parser.add_argument('--output-dir', default='data')
    argument_parser.add_argument('--train-size', default=0.7, type=float)
    args = argument_parser.parse_args()

    sub_dirs = get_subdirs(args.input_dir)

    for class_dir in sub_dirs:
        file_paths = get_all_files(class_dir)

        # Remove the folder above the dataset
        class_dir = class_dir.split('/')[-1]
        train_files, test_files = train_test_split(file_paths, train_size=args.train_size)
        move_files(train_files, args.output_dir + '/train/' + class_dir)
        move_files(test_files, args.output_dir + '/test/' + class_dir)



if __name__ == '__main__':
    main()
