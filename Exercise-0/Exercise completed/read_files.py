import pandas as pd


def read_files(filename):
    """
    Reads the given file and returns a grouped pandas object, grouped by "pedestrianId".

    :param filename:
    :return:
    """
    data = pd.read_csv(filename, delimiter=" ")
    return data.groupby("pedestrianId")
