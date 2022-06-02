import json

from playground.color import Color
from playground.geometry import Vector, VectorRef, Transform
from playground.shapes import Circle, Rectangle
from noise import *

for i in range(0, 360, 360 // 12):
    center = Vector.polar(i, 300)  # randint(-1500, 1500), randint(-1500, 1500)
    color = Color(h=i + 21, opacity=170)  # randint(0, 360))
    print(color.hex)
    # color.lightness /= 4
    Circle(center=center, radius=75, fill=color)

for i in range(5, 100, 5):
    for j in range(0, 130, 5):
        color = Color(J=i, C=j, h=291, clipped=False)
        Circle(center=(500 + j * 23, i * 23), radius=50, fill=color)

color = Color(h=a)
Circle(center=(0, -500), radius=75, fill=color)
color2 = Color(h=a + 120)
Circle(center=(120, -500), radius=75, fill=color2)
color3 = Color(h=a - 120)
Circle(center=(-120, -500), radius=75, fill=color3)
color4 = Color(h=a + 180)
Circle(center=(0, -620), radius=75, fill=color4)

print(Color("#ffff00").hue)

clr = Color("firebrick2")
hq = False
# hq = True
for j in []:  # range(1, 90, 10):
    r = j * 5
    for i in range(j):
        hue = 360 * i / j
        if hq:
            clr = Color(H=hue * 400 / 360, clipped=False)
        else:
            clr = Color(a / 3.6, j, hue * 400 / 360, mode="JCH", clipped=False)
        Circle((r * cos(radians(hue)), r * sin(radians(hue))), stroke=None, radius=10, fill=clr)


# kuler = json.load(open('kuler-themes.json'))
# themes = [['#'+s['hex'] for s in k['swatches']] for k in kuler if k['rating']['overall'] > 4. and k['rating']['count'] > 30]
for row, t in []:  # enumerate(themes):
    # colors = []
    Circle(center=(-300, -800 - row * 300), radius=5, fill="red")
    for i in range(5):
        color = Color(t[i])
        # colors.append(color)
        Circle(center=(i * 150, -800 - row * 300), radius=75, fill=color)
        if color.chroma > 10:
            Circle(
                center=Vector(-300, -800 - row * 300) + Vector.polar(color.hue, 2 * color.chroma), radius=10, fill=color
            )

import pickle

# colorhunt = pickle.load(open('colorhunt-top40.pickle', 'rb'))
# for row, t in enumerate(colorhunt):
#    #colors = []
#    Circle(center=(1200-300, -800 - row*300), radius=5, fill='red')
#    for i in range(4):
#        color = Color(t['colors'][i])
#        #colors.append(color)
#        Circle(center=(1200+i*150, -800 - row*300), radius=75, fill=color)
#        Circle(center=Vector(1200-300, -800 - row*300) + Vector.polar(color.hue, 2*color.chroma), radius=10, fill=color)
