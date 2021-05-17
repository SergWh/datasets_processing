from enum import Enum
import csv

START_STR = 'start'
END_STR = 'end'
PITCH_STR = 'pitch'
VELOCITY_STR = 'velocity'
STYLE_STR = 'style'
CSV_HEADER = [START_STR, END_STR, PITCH_STR, VELOCITY_STR, STYLE_STR]


class Style(Enum):
    NORMAL = 1
    MUTE = 2
    VIBRATO = 3
    BEND = 4
    PULL_OFF = 5
    HAMMER_ON = 6
    SLIDE = 7

    @staticmethod
    def parse(label):
        if label == 'normal':
            return Style.NORMAL
        elif label == 'mute':
            return Style.MUTE
        elif label == 'vibrato':
            return Style.VIBRATO
        elif label == 'bend':
            return Style.BEND
        elif label == 'pull-off':
            return Style.PULL_OFF
        elif label == 'hammer-on':
            return Style.HAMMER_ON
        elif label == 'slide':
            return Style.SLIDE
        else:
            raise NotImplementedError

    def to_str(self):
        if self == Style.NORMAL:
            return 'normal'
        elif self == Style.MUTE:
            return 'mute'
        elif self == Style.VIBRATO:
            return 'vibrato'
        elif self == Style.BEND:
            return 'bend'
        elif self == Style.PULL_OFF:
            return 'pull-off'
        elif self == Style.HAMMER_ON:
            return 'hammer-on'
        elif self == Style.SLIDE:
            return 'slide'


class Note:
    start = 0.0
    end = 0.0
    style = Style.NORMAL
    velocity = 0
    pitch = 0.0

    def __init__(self, start, end, style, velocity, pitch):
        self.start = start
        self.end = end
        self.style = style
        self.velocity = velocity
        self.pitch = pitch

    @staticmethod
    def parse_csv_row(row):
        return Note(
            start=float(row[START_STR]),
            end=float(row[END_STR]),
            pitch=int(row[PITCH_STR]),
            velocity=int(row[VELOCITY_STR]),
            style=Style.parse(row[STYLE_STR])
        )

    def __str__(self):
        return f'start={self.start},end={self.end},style={self.style},velo={self.velocity},pitch={self.pitch}'

    def __repr__(self):
        return self.__str__()
