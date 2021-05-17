import glob
import os

import soundfile as sf


def check(filename):
    ob = sf.SoundFile(filename)
    print('Sample rate: {}'.format(ob.samplerate))
    print('Channels: {}'.format(ob.channels))
    print('Subtype: {}'.format(ob.subtype))


def convert_file(filename, n):
    paths = filename.split('/')
    new_folder = 'IDMT-SMT-32'
    a = paths[:len(paths) - n - 1]
    a.append(new_folder)
    a.extend(paths[-n:])
    new_name = '/'.join(a)
    print(new_name)
    data, samplerate = sf.read(filename)
    os.makedirs(os.path.dirname(new_name), exist_ok=True)
    sf.write(new_name, data, samplerate, subtype='FLOAT')


# name = '/home/moby/PycharmProjects/data/IDMT-SMT-GUITAR_V2/'
# files = glob.glob(name + '/*/*/audio/*.wav', recursive=True)
#
# for file in files:
#     convert_file(file, 4)
# print("NEXT BATCH")
# files = glob.glob(name + '/*/audio/*.wav', recursive=True)
#
# for file in files:
#     convert_file(file, 3)
filename = '/home/moby/Downloads/rhcp_home2.wav'
check(filename)
data, samplerate = sf.read(filename)
sf.write(filename.split('.')[0] + '2.wav', data, samplerate, subtype='FLOAT')
check('/home/moby/Downloads/rhcp_resampled2.wav')

# import subprocess
#
# subprocess.call(['ffmpeg', '-i', filename,
#                  filename.split('.')[0] + '2.wav'])
