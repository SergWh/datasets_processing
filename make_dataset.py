import glob
import pandas as pd

export_path = 'dataset.csv'


def make_data(export_path):
    files = []
    path = '/home/moby/PycharmProjects/data/gpt/csv2/'
    files.extend(glob.glob(path + '*/*/*.csv'))
    path = '/home/moby/PycharmProjects/data/IDMT-SMT-32/'
    files.extend(glob.glob(path + '*/csv2/*.csv'))
    path = '/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset1/'
    files.extend(glob.glob(path + '*/csv2/*.csv'))
    combined_csv = pd.concat([pd.read_csv(f, header=None, skiprows=1) for f in files])
    combined_csv.to_csv(export_path)


def get_data(path, label_row=5, data_rows=tuple(range(134))[6:134]):
    data = pd.read_csv(path, header=None, usecols=data_rows, skiprows=1).to_numpy()
    labels = pd.read_csv(path, header=None, usecols=[label_row], skiprows=1).to_numpy()
    return data, labels
