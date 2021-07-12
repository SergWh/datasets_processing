from enum import Enum

START_STR = 'start'
END_STR = 'end'
PITCH_STR = 'pitch'
STYLE_STR = 'style'
CSV_HEADER = [START_STR, END_STR, PITCH_STR, STYLE_STR]


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
            return 'normal'

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
    pitch = 0.0

    def __init__(self, start, end, style, pitch):
        self.start = start
        self.end = end
        self.style = style
        self.pitch = pitch

    @staticmethod
    def parse_csv_row(row):
        pitch = (row[PITCH_STR])
        if pitch == 'nan':
            pitch = -1
        else:
            pitch = int(float(pitch))
        return Note(
            start=float(row[START_STR]),
            end=float(row[END_STR]),
            pitch=pitch,
            style=Style.parse(row[STYLE_STR])
        )

    def __str__(self):
        return f'start={self.start},end={self.end},style={self.style},pitch={self.pitch}'

    def __repr__(self):
        return self.__str__()
