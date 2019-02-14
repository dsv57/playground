#from math import *
#from turtle import *

import numpy as np
from numpy import sin, cos, arctan2 as atan2, \
                  sqrt, ceil, floor, degrees, radians, log, pi, exp, transpose
from colorio import CIELAB, CAM16UCS, JzAzBz, SrgbLinear

L_A = 64 / pi / 5
srgb = SrgbLinear()
lab = CIELAB()
cam16ucs = CAM16UCS(0.69, 20, L_A)
jzazbz = JzAzBz()

NEUTRAL = '#262626'

def lab_to_srgb(color):
    return transpose(srgb.to_srgb1(srgb.from_xyz100(lab.to_xyz100(transpose(color)))))

def jzazbz_to_srgb(color):
    return transpose(srgb.to_srgb1(srgb.from_xyz100(jzazbz.to_xyz100(transpose(color)))))

def srgb_to_lab(color):
    return transpose(lab.from_xyz100(srgb.to_xyz100(srgb.from_srgb1(transpose(color)))))

def srgb_to_jzazbz(color):
    return transpose(jzazbz.from_xyz100(srgb.to_xyz100(srgb.from_srgb1(transpose(color)))))

def lab_to_jzazbz(color):
    return transpose(jzazbz.from_xyz100(lab.to_xyz100(transpose(color))))

def lab_to_cam16ucs(color):
    return transpose(cam16ucs.from_xyz100(lab.to_xyz100(transpose(color))))

def jzazbz_to_lab(color):
    return transpose(lab.from_xyz100(jzazbz.to_xyz100(transpose(color))))

def cam16ucs_to_srgb(color):
    return transpose(srgb.to_srgb1(srgb.from_xyz100(cam16ucs.to_xyz100(transpose(color)))))

def srgb_to_cam16ucs(color):
    return transpose(cam16ucs.from_xyz100(srgb.to_xyz100(srgb.from_srgb1(transpose(color)))))

def _square_plot(l, k1, k2, conv, d=30):
    for x in range(-300,301,d):
        for y in range(-300,301,d):
            goto(x, y)
            c = conv(l, x * k1, y * k2)
            if np.max(c) > 1.0 or np.min(c) < 0.0 or np.isnan(c).any():
                tc = NEUTRAL
            else:
                tc = tuple(map(float, c))
            dot(d, tc)
    update()

def plot_lab(l=73, r=30):
    _square_plot(l, 1/4, 1/4, lab_to_srgb, r)

def plot_cam16ucs(j=75, r=30):
    _square_plot(j, 1/10, 1/10, cam16ucs_to_srgb, r)

def plot_jzazbz(jz=0.12, r=30):
    _square_plot(jz, 1/3000, 1/3000, jzazbz_to_srgb, r)

def _diag_plot(l, k1, k2, conv, d=30):
    x_step = int(round(tan(pi/3) * d/2))+1
    for x in range(-300,301,x_step):
        offset = (x//x_step) % 2 * d//2
        for y in range(-300+offset, 301+offset,d+1):
            goto(x, y)
            c = conv(l, x * k1, y * k2)
            if np.max(c) > 1.0 or np.min(c) < 0.0 or np.isnan(c).any():
                tc = NEUTRAL
            else:
                tc = tuple(map(float, c))
            dot(d, tc)
    update()

def dplot_lab(l=73, d=30):
    _diag_plot(l, 1/4, 1/4, lab_to_srgb, d)

def dplot_cam16ucs(j=75, d=30):
    _diag_plot(j, 1/10, 1/10, cam16ucs_to_srgb, d)

def dplot_jzazbz(jz=0.12, d=30):
    _diag_plot(jz, 1/3000, 1/3000, jzazbz_to_srgb, d)

def _radial_plot(l, k1, k2, conv, big=True):
    for i in range(1,40,3 if big else 2):
        r = 10 * i
        points = i*(2 if big else 3)
        for j in range(points):
            angle = 2*pi * j/points
            x = r*cos(angle)
            y = r*sin(angle)
            goto(x, y)
            c = conv(l, x * k1, y * k2)
            if np.max(c) > 1.0 or np.min(c) < 0.0 or np.isnan(c).any():
                tc = NEUTRAL
            else:
                tc = tuple(map(float, c))
            dot(29 if big else 19, tc)
    update()

def rplot_lab(l=73, big=True):
    _radial_plot(l, 1/4, 1/4, lab_to_srgb, big)

def rplot_jzazbz(jz=0.12, big=True):
    _radial_plot(jz, 1/3000, 1/3000, jzazbz_to_srgb, big)

def rplot_cam16ucs(j=75, big=True):
    _radial_plot(j, 1/10, 1/10, cam16ucs_to_srgb, big)

def to_radians(a):
    return a / 180 * pi

def bright_jzazbz(jz, angle, steps=100):
    prev_c = None
    r = 0
    for i in range(50000):
        r += 0.1 / steps
        x = r*cos(to_radians(angle))
        y = r*sin(to_radians(angle))
        c = jzazbz_to_srgb((jz, x, y))
        if np.max(c) > 1.0 or np.min(c) < 0.0 or np.isnan(c).any():
            print(i)
            return prev_c
        else:
            prev_c = c

def bright_cam16ucs(j, angle, steps=100):
    prev_c = None
    r = 0
    for i in range(50000):
        r += 30 / steps
        a = r*cos(to_radians(angle))
        b = r*sin(to_radians(angle))
        c = cam16ucs_to_srgb((j, a, b))
        if np.max(c) > 1.0 or np.min(c) < 0.0 or np.isnan(c).any():
            #print(i)
            return prev_c
        else:
            prev_c = c

# !!!
def show_jzazbz_hues():
    for j in range(0,11):
        steps = 32
        if j == 0: steps = 4
        if j == 1: steps = 8
        if j == 2: steps = 16
        for i in range(steps):
            r = 20 + 50*j
            angle = 360 * i/steps
            goto(r*cos(to_radians(angle)), r*sin(to_radians(angle)))
            dot(50, tuple(map(float,bright_jzazbz(0.05 + j/100, angle))))
            update()

def show_cam16ucs_hues():
    for j in range(1,10):
        steps = 32
        if j == 0: steps = 4
        if j == 1: steps = 8
        if j == 2: steps = 16
        for i in range(steps):
            r = 20 + 50*j
            angle = 360 * i/steps
            goto(r*cos(to_radians(angle)), r*sin(to_radians(angle)))
            dot(50, tuple(map(float,bright_cam16ucs(j*10, angle))))
            update()


#clear()
#for i in range(1,40,3 if big else 2):
#    r = 10 * i
#    points = i*(2 if big else 3)
#    for j in range(points):
#        angle = 2*pi * j/points
#        x = r*cos(angle)
#        y = r*sin(angle)
#        goto(x, y)
#        dot(29 if big else 19, tuple(map(float,bright_jzazbz(0.04 + i/400, angle))))
#    update()

#x_step = int(round(tan(pi/3) * d/2))+1
#for x in range(-300,301,x_step):
#    offset = (x//x_step) % 2 * d//2
#    for y in range(-300+offset, 301+offset,d+1):
#        goto(x, y)
#        angle = atan2(y, x)
#        r = sqrt(x*x + y*y)
#        dot(d, tuple(map(float,bright_jzazbz(0.04 + r/4000, angle))))
#    update()

# NCS
#ncs = eval(open('/home/user/edu/colour/ncs-final-int').read())
#colormode(255)
#for cc in [x for x in ncs if x[0].split('-')[0] == '2050']:
#    c = lab_to_cam16ucs(*cc[1][-1])
#    goto(c[1] * 10, c[2] * 10)
#    dot(20, 'black')
#    dot(10, cc[1][0])
# chrom = 20
# for blackness in range(5, 80+1, 5):
#     for cc in [x for x in ncs if x[0].split('-')[0] == f'{blackness:02}{chrom}']:
#         a = int(cc[0].split('-')[1][1:3] or '0')
#         if a not in [0, 30, 50, 70]:
#             continue
#         c = lab_to_cam16ucs(cc[1][-1])
#         angle = atan2(c[2], c[1])
#         r = c[0]
#         goto(r*cos(angle), r*sin(angle))
#         dot(6, [c/255 for c in cc[1][0]])


#colormode(255)
#for h in hues[::2]:
#    for cc in ncs[::]:
#        if cc[0].split('-')[1] == h:
#            c = lab_to_cam16ucs(*cc[1][-1])
#            goto(c[1] * 10, c[2] * 10)
#            #dot(20, 'black')
#            dot(5, cc[1][0])
#    update()

# hues='B,B10G,B20G,B30G,B40G,B50G,B60G,B70G,B80G,B90G,G,G10Y,G20Y,G30Y,G40Y,G50Y,G60Y,G70Y,G80Y,G90Y,N,R,R10B,R20B,R30B,R40B,R50B,R60B,R70B,R80B,R90B,Y,Y10R,Y20R,Y30R,Y40R,Y50R,Y60R,Y70R,Y80R,Y90R'.split(',')

def show_kluwer_themes(themes, rows=7, r=50):
    row = 1
    for t in themes:
        for s in t['swatches']:
            color = '#' + s['hex']
            dot(r, color)
            write(color, align='center')
            forward(r)
        backward(r*len(t['swatches']))
        if row < rows:
            right(90)
            forward(r)
            left(90)
            row = row + 1
        else:
            forward(r*6)
            left(90)
            forward((rows - 1) * r)
            right(90)
            row = 1


#import matplotlib.pyplot as plt
#image = plt.imread('/usr/share/matplotlib/sample_data/grace_hopper.png')
#image.shape == (600, 512, 3)
#image_c16 = cam16ucs.from_xyz100(srgb.to_xyz100(srgb.from_srgb1(image.reshape((image.shape[0] * image.shape[1], 3)).T)))
#pass
#image_sr = srgb.to_srgb1(srgb.from_xyz100(cam16ucs.to_xyz100(image_c16)))
#plt.imshow(image_sr.T.reshape(image.shape)); plt.show()
