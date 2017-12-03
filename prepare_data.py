"""
Short script to move files to one folder
"""

import os
import argparse
import shutil

from util import get_all_files, parse_file_path, \
    is_image


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--input-dir')
    argument_parser.add_argument('--output-dir', default='train')

    args = argument_parser.parse_args()

    file_paths = get_all_files(args.input_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i, file_path in enumerate(file_paths):
        file_name = parse_file_path(file_path)

        # move the file to the output dir if it's a JPEG
        if '_' in file_name and not file_name == '256_ObjectCategories':
            output_file_path = "{}/{}".format(args.output_dir, file_name)
            shutil.move(file_path, output_file_path)
            print("{}: Moved file {}!".format(i, file_name))

if __name__ == '__main__':
    main()
