import os

def get_all_files(directory):
    """
    Takes in a directory and returns a list of all the
    files in a dir ands its subdirs
    :return:
    """

    files = []

    for path, subdirs, dir_files in os.walk(directory):
        for name in dir_files:
            files.append(os.path.join(path, name))

    return files


def parse_file_path(file_path):
    """
    Parses a file path and returns the file name
    i.e., the last element in the list when the string
    is seperated by /
    :param file_path:
    :return:
    """

    return file_path.split('/')[-1]

def is_image(image_path):
    """
    Simple function that splits image file path
    and checks if the last element is jpg

    :param image_path:
    :return: a boolean, True if it's a JPEG, False otherwise
    """

    return 'jpg' == image_path.split('.')[-1]


def get_subdirs(directory):
    subdirs = []

    for path in os.listdir(directory):
        path = os.path.join(directory, path)

        if os.path.isdir(path):
            subdirs.append(path)

    return subdirs
