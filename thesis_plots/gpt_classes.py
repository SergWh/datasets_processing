import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

img_path = '/home/moby/PycharmProjects/datasets_processing/figures/'
plt.rcParams.update({

    'font.family': 'serif',
    'font.sans-serif': ['Times'],
    'text.latex.preamble':
        r'\usepackage[T2A]{fontenc}'
        r'\usepackage[utf8]{inputenc}'
})
classes = [
    'Normal (Без спец. техники)',
    'Muting (Пиццикато)',
    'Vibrato (Вибрато)',
    'Pull-off (Нисх. легато)',
    'Hammer-on (Восх. легато)',
    'Sliding (Глиссандо)',
    'Bending (Бенд)'
]
vals = np.array([
    2009,
    385,
    637,
    525,
    581,
    1162,
    1281
])


def absolute_value(val):
    a = numpy.round(val / 100. * vals.sum(), 0)
    return int(a)


fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.pie(vals, labels=classes, autopct=absolute_value,
        shadow=False, startangle=90,
        textprops={'fontsize': 15}
        )
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
# fig1.set_size_inches(18.5, 10.5)
# plt.tight_layout()
plt.savefig(img_path + "gpt_pie" + '.pdf')
plt.show()
