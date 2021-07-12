import csv

import numpy as np

from note_utils import Note


def read_notes(name):
    with open(name, 'r') as file:
        reader = csv.DictReader(file)
        return np.array(list(map(Note.parse_csv_row, list(reader))))


def read_notes_non_np(name):
    # print(name)
    with open(name, 'r') as file:
        reader = csv.DictReader(file)
        return list(map(Note.parse_csv_row, list(reader)))
