import glob
import xml.etree.ElementTree as ET
import os
import csv

name = '/home/moby/PycharmProjects/data/IDMT-SMT-GUITAR_V2/'
f_name = 'dataset2/annotation/AR_A_fret_0-20.xml'


def note_array(event, velo=70):
    start = float(event.find('onsetSec').text)
    end = float(event.find('offsetSec').text)
    pitch = int(event.find('pitch').text)
    excit = event.find('excitationStyle').text
    express = event.find('expressionStyle').text
    style = 'normal'
    if express == 'BE':
        style = 'bend'
    elif express == 'VI':
        style = 'vibrato'
    elif express == 'SL':
        style = 'slide'
    elif express == 'NO' and excit == 'MU':
        style = 'mute'
    return [start, end, pitch, velo, style]


def parse_file(filename):
    header = ('start', 'end', 'pitch', 'velocity', 'style')
    tree = ET.parse(filename)
    root = tree.getroot()
    events = root[1].findall('event')
    if len(events) == 0:
        return
    split_path = filename.split('.')[0].split('/')
    new_name = '/'.join(split_path[:len(split_path) - 2]) + '/csv/' + split_path[-1] + '.csv'
    print(new_name)
    os.makedirs(os.path.dirname(new_name), exist_ok=True)
    with open(new_name, 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        for event in events:
            writer.writerow(note_array(event))


files = []
files.extend(glob.glob(name + '/*/annotation/*.xml', recursive=True))
files.extend(glob.glob(name + '/*/*/annotation/*.xml', recursive=True))
for file in files:
    parse_file(file)
