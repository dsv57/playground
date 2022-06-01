from collections import defaultdict
from functools import lru_cache
from copy import copy, deepcopy
import inspect

import numpy as np
from numpy import sin, cos, arctan2 as atan2, \
                  sqrt, ceil, floor, degrees, radians, log, pi, exp, transpose
from numpy.polynomial import polynomial
from scipy.spatial import KDTree
from colorio import CIELAB, CAM16UCS, CAM16, JzAzBz, SrgbLinear
from colorio.illuminants import whitepoints_cie1931

from named_colors import COLORS

from playground.utils import KeepWeakRefs


__all__ = ['Color']


_dims = {
    'lightness': ('J', 0, 0),
    'brightness': ('Q', 6, 0),
    'chroma': ('C', 1, 1),
    'colorfulness': ('M', 4, 1),
    'saturation': ('s', 5, 1),
    'hue_quadrature': ('H', 2, 2),
    'hue': ('h', 3, 2),
}
_ltrs = list('JCHhMsQ') # 'JQCMsHh'
_ltr2pos = {ltr: pos for ltr, pos, ord in _dims.values()}
_ltr2ord = {ltr: ord for ltr, pos, ord in _dims.values()}

_srgb_linear = SrgbLinear()
_cielab = CIELAB()

INV_COLORS = {v: k for k, v in COLORS.items()}

_hues = np.linspace(0, 360-1, 360, dtype='int')
_hqs = np.linspace(0, 400-1, 200, dtype='int')

_bright_hues_avg_20_20 = np.array([
       [ 48.796875  ,  48.75      ,  48.4921875 ,  48.375     ,
         48.1875    ,  48.09375   ,  48.        ,  47.8125    ,
         47.71875   ,  47.625     ,  47.484375  ,  47.4375    ,
         47.296875  ,  47.25      ,  47.25      ,  47.0625    ,
         46.921875  ,  46.875     ,  46.78125   ,  46.69921875,
         46.59375   ,  46.546875  ,  46.5       ,  46.5       ,
         46.359375  ,  46.3125    ,  46.23046875,  46.171875  ,
         46.546875  ,  47.25      ,  48.        ,  48.52148438,
         49.125     ,  49.734375  ,  50.34375   ,  51.        ,
         51.46875   ,  52.03125   ,  52.59375   ,  53.25      ,
         53.671875  ,  54.1875    ,  54.75      ,  55.125     ,
         55.6875    ,  56.25      ,  56.671875  ,  57.1875    ,
         57.75      ,  58.125     ,  58.5       ,  59.0625    ,
         59.484375  ,  60.        ,  60.421875  ,  60.84375   ,
         61.3359375 ,  61.78125   ,  62.25      ,  62.671875  ,
         63.09375   ,  63.5625    ,  63.9375    ,  64.5       ,
         64.875     ,  65.25      ,  65.8125    ,  66.2578125 ,
         66.75      ,  67.16015625,  67.59375   ,  68.07421875,
         68.4375    ,  69.        ,  69.375     ,  69.9375    ,
         70.3125    ,  70.875     ,  71.34375   ,  71.8125    ,
         72.375     ,  72.84375   ,  73.359375  ,  73.8984375 ,
         74.4375    ,  75.        ,  75.46875   ,  76.03125   ,
         76.60546875,  77.25      ,  77.71875   ,  78.375     ,
         78.75      ,  79.59375   ,  80.25      ,  80.8125    ,
         81.5625    ,  82.21875   ,  82.875     ,  83.671875  ,
         84.4453125 ,  85.21875   ,  86.015625  ,  86.84765625,
         87.5625    ,  88.59375   ,  89.4375    ,  90.375     ,
         91.5       ,  92.53125   ,  93.609375  ,  94.59375   ,
         94.21875   ,  93.75      ,  93.1875    ,  92.859375  ,
         92.4375    ,  91.96875   ,  91.5       ,  90.9375    ,
         90.5625    ,  90.        ,  89.6953125 ,  89.25      ,
         88.6875    ,  88.265625  ,  87.75      ,  87.28125   ,
         86.8359375 ,  86.25      ,  85.78125   ,  85.3359375 ,
         84.8203125 ,  84.28125   ,  83.625     ,  83.25      ,
         82.7109375 ,  82.1484375 ,  81.5859375 ,  81.        ,
         80.4375    ,  79.83984375,  79.21875   ,  79.32421875,
         79.5       ,  79.5       ,  79.734375  ,  79.875     ,
         80.015625  ,  80.15625   ,  80.25      ,  80.390625  ,
         80.53125   ,  80.625     ,  80.7890625 ,  80.8125    ,
         81.03515625,  81.        ,  81.28125   ,  81.375     ,
         81.375     ,  81.5625    ,  81.75      ,  81.84375   ,
         81.9375    ,  82.0546875 ,  82.16015625,  82.265625  ,
         82.37109375,  82.5       ,  82.5703125 ,  82.6875    ,
         82.78125   ,  82.875     ,  82.96875   ,  83.0625    ,
         83.15625   ,  83.25      ,  83.34375   ,  83.4375    ,
         83.4375    ,  83.625     ,  83.625     ,  83.8125    ,
         84.        ,  84.0234375 ,  84.1171875 ,  84.1875    ,
         84.3046875 ,  84.375     ,  84.375     ,  84.375     ,
         84.75      ,  84.75      ,  84.84375   ,  84.984375  ,
         84.9375    ,  84.75      ,  84.        ,  83.25      ,
         82.640625  ,  81.9375    ,  81.28125   ,  80.625     ,
         79.96875   ,  79.3125    ,  78.75      ,  78.        ,
         77.4375    ,  76.875     ,  76.125     ,  75.75      ,
         75.09375   ,  74.53125   ,  73.9921875 ,  73.5       ,
         72.84375   ,  72.3046875 ,  71.625     ,  71.25      ,
         70.5       ,  70.125     ,  69.5625    ,  69.        ,
         68.484375  ,  67.875     ,  67.5       ,  66.75      ,
         66.375     ,  65.8125    ,  65.25      ,  64.6875    ,
         64.125     ,  63.75      ,  63.        ,  62.625     ,
         62.0625    ,  61.5       ,  60.9375    ,  60.375     ,
         60.        ,  59.25      ,  58.6875    ,  58.125     ,
         57.75      ,  57.        ,  56.34375   ,  55.78125   ,
         55.125     ,  54.75      ,  54.        ,  53.34375   ,
         52.6875    ,  52.125     ,  51.421875  ,  51.        ,
         50.25      ,  49.5       ,  48.75      ,  48.01171875,
         47.25      ,  46.546875  ,  45.796875  ,  45.        ,
         44.25      ,  43.5       ,  42.75      ,  42.        ,
         41.25      ,  40.125     ,  39.1875    ,  38.25      ,
         37.5       ,  36.375     ,  35.25      ,  34.5       ,
         33.375     ,  32.25      ,  31.125     ,  30.        ,
         30.        ,  30.        ,  30.        ,  30.        ,
         30.        ,  30.        ,  30.        ,  30.        ,
         30.        ,  30.        ,  30.        ,  30.        ,
         30.        ,  30.        ,  30.        ,  30.        ,
         30.        ,  30.75      ,  30.7734375 ,  31.5       ,
         31.6875    ,  32.25      ,  32.625     ,  33.        ,
         33.75      ,  33.9375    ,  34.5       ,  34.875     ,
         35.34375   ,  36.        ,  36.375     ,  36.9375    ,
         37.5       ,  38.25      ,  38.625     ,  39.19335938,
         39.75      ,  40.5       ,  41.25      ,  42.        ,
         42.375     ,  43.125     ,  43.875     ,  44.625     ,
         45.375     ,  46.125     ,  46.875     ,  47.71875   ,
         48.75      ,  49.5       ,  50.34375   ,  51.375     ,
         52.22460938,  53.25      ,  54.28125   ,  54.84375   ,
         54.515625  ,  54.1875    ,  53.8359375 ,  53.53125   ,
         53.25      ,  52.8984375 ,  52.59375   ,  52.3359375 ,
         52.125     ,  51.80273438,  51.5625    ,  51.28125   ,
         51.0703125 ,  50.84179688,  50.625     ,  50.4375    ,
         50.25      ,  50.00390625,  49.875     ,  49.59375   ,
         49.5       ,  49.265625  ,  49.125     ,  48.9375    ],
       [102.17421125, 101.80384088, 102.29766804, 102.32510288,
        102.44855967, 102.66803841, 102.69547325, 102.94238683,
        103.18930041, 103.43621399, 103.68312757, 103.89346136,
        104.30041152, 104.42386831, 104.38728852, 105.24843774,
        105.781893  , 106.15226337, 106.73296754, 107.26337449,
        107.75720165, 108.3744856 , 108.86831276, 109.19905502,
        110.31016613, 110.93964335, 111.80384088, 112.57201646,
        110.8436214 , 107.46964386, 103.80658436, 101.58436214,
         98.86831276,  96.37174211,  94.0260631 ,  91.46090535,
         89.73251029,  87.75720165,  86.00137174,  83.80658436,
         82.65584515,  81.09053498,  79.48559671,  78.25102881,
         77.01646091,  75.63100137,  74.67078189,  73.53223594,
         72.20164609,  71.58436214,  70.59670782,  69.73251029,
         68.96433471,  68.12757202,  67.48285322,  66.76954733,
         66.15226337,  65.53497942,  64.91769547,  64.42386831,
         63.90260631,  63.43621399,  62.94238683,  62.44855967,
         62.20164609,  61.83127572,  61.58436214,  61.31001372,
         60.8436214 ,  60.81618656,  60.59670782,  60.43667124,
         60.22633745,  60.10288066,  59.97942387,  59.95198903,
         59.85596708,  59.85596708,  59.85596708,  59.85596708,
         59.93979576,  59.97942387,  60.10288066,  60.22633745,
         60.3223594 ,  60.34979424,  60.72016461,  60.93964335,
         61.18655693,  61.18655693,  61.70781893,  62.04160951,
         62.32510288,  62.78235025,  63.15272062,  63.55967078,
         64.05349794,  64.5473251 ,  65.04115226,  65.63100137,
         66.23914037,  66.86556927,  67.51028807,  68.22359396,
         68.86831276,  69.73251029,  70.47325103,  71.33744856,
         72.17421125,  73.28532236,  74.30041152,  75.28806584,
         75.781893  ,  76.27572016,  76.76954733,  77.38683128,
         77.97668038,  78.59396433,  79.23868313,  79.85596708,
         80.59670782,  81.33744856,  82.20164609,  82.81893004,
         83.80658436,  84.76680384,  85.65843621,  86.64609053,
         87.7297668 ,  88.74485597,  89.85596708,  91.09053498,
         92.29766804,  93.55967078,  94.79423868,  96.27572016,
         97.7297668 ,  99.23868313, 100.81618656, 102.44855967,
        104.17695473, 105.99222679, 107.85322359, 104.88111568,
        101.5569273 ,  98.86831276,  96.27572016,  93.80658436,
         91.58436214,  89.45816187,  87.48285322,  85.65843621,
         83.93004115,  82.32510288,  80.8436214 ,  79.36213992,
         78.10013717,  76.76954733,  75.65843621,  74.5473251 ,
         73.43621399,  72.44855967,  71.46090535,  70.69272977,
         69.85596708,  69.0877915 ,  68.34705075,  67.63374486,
         66.97988112,  66.24828532,  65.75445816,  65.16460905,
         64.64334705,  64.14951989,  63.68312757,  63.18930041,
         62.81893004,  62.42112483,  62.05075446,  61.70781893,
         61.33744856,  61.06310014,  60.72016461,  60.47325103,
         60.07544582,  60.06630087,  59.85596708,  59.6090535 ,
         59.48559671,  59.32556013,  59.11522634,  58.99176955,
         58.74485597,  58.84087791,  58.74485597,  58.71742112,
         58.62139918,  58.47050754,  58.12757202,  57.88065844,
         57.63374486,  57.38683128,  57.1399177 ,  56.89300412,
         56.74211248,  56.52263374,  56.36259717,  56.15226337,
         56.02880658,  55.90534979,  55.781893  ,  55.65843621,
         55.63100137,  55.53497942,  55.50754458,  55.28806584,
         55.41152263,  55.38408779,  55.28806584,  55.260631  ,
         55.28806584,  55.28806584,  55.41152263,  55.41152263,
         55.50754458,  55.53497942,  55.50754458,  55.65843621,
         55.781893  ,  55.90534979,  56.02880658,  56.15226337,
         56.27572016,  56.37174211,  56.64609053,  56.76954733,
         57.01646091,  57.23593964,  57.48285322,  57.7297668 ,
         57.75720165,  58.25102881,  58.49794239,  58.82766855,
         58.86831276,  59.36213992,  59.73251029,  60.10288066,
         60.47325103,  60.69272977,  61.21399177,  61.70781893,
         62.0781893 ,  62.54458162,  63.06584362,  63.28532236,
         63.90260631,  64.51989026,  65.13717421,  65.75445816,
         66.27572016,  66.97988112,  67.60631001,  68.25102881,
         68.99176955,  69.6090535 ,  70.3223594 ,  70.96707819,
         71.70781893,  72.7914952 ,  73.68312757,  74.5473251 ,
         75.28806584,  76.37174211,  77.38683128,  78.12757202,
         79.23868313,  80.3223594 ,  81.33744856,  82.42112483,
         82.54458162,  82.69547325,  82.81893004,  83.03840878,
         83.18930041,  83.43621399,  83.68312757,  84.01386984,
         84.30041152,  84.5473251 ,  84.91769547,  85.28806584,
         85.65843621,  86.15226337,  86.52263374,  87.01646091,
         87.51028807,  87.38683128,  87.96753544,  87.85322359,
         88.25102881,  88.3744856 ,  88.70827618,  88.99176955,
         88.99176955,  89.58161866,  89.73251029,  90.22633745,
         90.59670782,  90.72016461,  91.31001372,  91.67123914,
         92.05075446,  92.20164609,  92.91495199,  93.40877915,
         93.80658436,  94.26383173,  94.5473251 ,  95.00457247,
         95.90534979,  96.48605396,  97.01646091,  97.60631001,
         98.25102881,  98.96433471,  99.73251029, 100.44581619,
        100.8436214 , 101.83127572, 102.78235025, 103.40877915,
        104.51074531, 105.28806584, 106.27572016, 106.76954733,
        106.39917695, 105.98917848, 105.63100137, 105.24843774,
        104.79423868, 104.6342021 , 104.30041152, 104.05349794,
        103.55967078, 103.55967078, 103.28532236, 103.06584362,
        102.94238683, 102.78235025, 102.57201646, 102.41197988,
        102.20164609, 102.29766804, 101.95473251, 102.0781893 ,
        101.95473251, 102.0781893 , 102.0781893 , 102.0781893 ]])


_bright_hqs_avg_20_20 = np.array([
       [ 46.59375   ,  46.5       ,  46.5       ,  46.3125    ,
         46.21875   ,  46.5703125 ,  47.625     ,  48.5859375 ,
         49.5       ,  50.4375    ,  51.3046875 ,  52.125     ,
         52.875     ,  53.7421875 ,  54.515625  ,  55.21875   ,
         55.96875   ,  56.6953125 ,  57.375     ,  57.9375    ,
         58.734375  ,  59.4375    ,  60.        ,  60.75      ,
         61.3125    ,  61.96875   ,  62.625     ,  63.1875    ,
         63.75      ,  64.5       ,  64.875     ,  65.625     ,
         66.234375  ,  66.84375   ,  67.5       ,  67.875     ,
         68.625     ,  69.234375  ,  69.8671875 ,  70.5       ,
         71.0625    ,  71.71875   ,  72.375     ,  72.9375    ,
         73.5       ,  74.296875  ,  75.        ,  75.5625    ,
         76.359375  ,  77.0625    ,  77.71875   ,  78.3984375 ,
         79.03125   ,  79.6875    ,  80.390625  ,  81.1171875 ,
         81.890625  ,  82.5       ,  83.484375  ,  84.375     ,
         85.265625  ,  86.25      ,  87.1875    ,  88.125     ,
         89.390625  ,  90.5625    ,  91.875     ,  93.1875    ,
         94.6875    ,  94.03125   ,  93.46875   ,  92.859375  ,
         92.25      ,  91.5       ,  90.9375    ,  90.375     ,
         89.671875  ,  88.9921875 ,  88.125     ,  87.5625    ,
         86.8125    ,  86.0625    ,  85.125     ,  84.421875  ,
         83.53125   ,  82.6875    ,  81.75      ,  80.765625  ,
         79.734375  ,  79.3125    ,  79.55859375,  79.6875    ,
         80.0625    ,  80.25      ,  80.53125   ,  80.8125    ,
         81.        ,  81.28125   ,  81.515625  ,  81.75      ,
         81.9375    ,  82.125     ,  82.23046875,  82.3125    ,
         82.5       ,  82.5       ,  82.734375  ,  82.875     ,
         82.9921875 ,  83.0625    ,  83.25      ,  83.34375   ,
         83.4375    ,  83.625     ,  83.71875   ,  83.8125    ,
         84.        ,  84.        ,  84.1875    ,  84.3984375 ,
         84.46875   ,  84.375     ,  84.80859375,  84.9375    ,
         85.03125   ,  84.375     ,  83.25      ,  82.3125    ,
         81.28125   ,  80.34375   ,  79.3125    ,  78.375     ,
         77.4375    ,  76.59375   ,  75.75      ,  74.625     ,
         73.875     ,  72.9375    ,  72.09375   ,  71.25      ,
         70.3125    ,  69.375     ,  68.53125   ,  67.5       ,
         66.75      ,  65.8125    ,  64.921875  ,  64.03125   ,
         63.        ,  62.25      ,  61.171875  ,  58.875     ,
         56.4375    ,  54.        ,  51.421875  ,  48.75      ,
         45.9375    ,  43.125     ,  39.84375   ,  36.75      ,
         33.        ,  30.        ,  30.        ,  30.        ,
         30.        ,  30.        ,  30.375     ,  31.5       ,
         33.        ,  34.5       ,  36.        ,  37.5       ,
         39.        ,  40.875     ,  42.75      ,  44.625     ,
         46.6875    ,  48.9375    ,  51.375     ,  54.        ,
         54.421875  ,  53.625     ,  52.875     ,  52.125     ,
         51.5625    ,  51.        ,  50.4375    ,  49.9921875 ,
         49.5       ,  49.171875  ,  48.75      ,  48.515625  ,
         48.22265625,  48.        ,  47.625     ,  47.484375  ,
         47.2734375 ,  47.0859375 ,  46.875     ,  46.6875    ],
       [107.85322359, 108.74485597, 109.23868313, 110.8436214 ,
        112.0781893 , 110.8436214 , 105.65843621, 101.31001372,
         97.26337449,  93.65569273,  90.44581619,  87.51028807,
         84.79423868,  82.42112483,  80.19890261,  78.12757202,
         76.27572016,  74.64334705,  73.06584362,  71.58436214,
         70.34979424,  69.0877915 ,  68.00411523,  66.89300412,
         66.12482853,  65.28806584,  64.42386831,  63.80658436,
         63.18930041,  62.42112483,  62.0781893 ,  61.70781893,
         61.31001372,  60.96707819,  60.47325103,  60.34979424,
         60.22633745,  60.07544582,  59.97942387,  59.85596708,
         59.85596708,  59.85596708,  59.95198903,  59.97942387,
         60.10288066,  60.3223594 ,  60.47325103,  60.72016461,
         61.06310014,  61.33744856,  61.70781893,  62.0781893 ,
         62.44855967,  62.81893004,  63.28532236,  63.77914952,
         64.30041152,  64.79423868,  65.49839963,  66.11568358,
         66.89300412,  67.51028807,  68.49794239,  69.36213992,
         70.44581619,  71.46090535,  72.69547325,  73.93004115,
         75.28806584,  75.90534979,  76.61865569,  77.35939643,
         78.12757202,  78.99176955,  79.97942387,  80.93964335,
         82.20164609,  83.43621399,  84.67078189,  86.15226337,
         87.75720165,  89.44901692,  91.21399177,  93.28532236,
         95.41152263,  97.75720165, 100.34979424, 103.18930041,
        106.27572016, 105.12802926,  99.73251029,  94.91769547,
         90.72016461,  87.01646091,  83.68312757,  80.69272977,
         78.00411523,  75.62185642,  73.43621399,  71.43347051,
         69.6090535 ,  68.62139918,  67.84407865,  66.98902606,
         66.15226337,  65.41152263,  64.79423868,  64.14951989,
         63.55967078,  62.94238683,  62.44855967,  61.95473251,
         61.46090535,  61.09053498,  60.72016461,  60.34979424,
         60.07544582,  59.73251029,  59.48559671,  59.32556013,
         59.11522634,  58.86831276,  58.84087791,  58.71742112,
         58.62139918,  58.25102881,  57.88065844,  57.51028807,
         57.1399177 ,  56.85642433,  56.52263374,  56.27572016,
         56.02880658,  55.87791495,  55.53497942,  55.53497942,
         55.41152263,  55.41152263,  55.37189453,  55.260631  ,
         55.38408779,  55.41152263,  55.50754458,  55.53497942,
         55.75445816,  55.90534979,  56.15226337,  56.39917695,
         56.64609053,  56.89300412,  57.38683128,  58.3744856 ,
         59.73251029,  61.21399177,  63.06584362,  65.04115226,
         67.38683128,  69.97942387,  73.03840878,  76.00137174,
         79.6090535 ,  82.53543667,  83.06584362,  83.93004115,
         84.91769547,  86.27572016,  87.51028807,  88.25102881,
         88.86831276,  89.6090535 ,  90.59670782,  91.80384088,
         93.18930041,  94.42386831,  95.90534979,  97.75720165,
         99.6090535 , 101.58436214, 103.68312757, 105.90534979,
        106.27572016, 105.25148605, 104.39643347, 103.80658436,
        103.18930041, 102.69547325, 102.44855967, 102.28852309,
        102.0781893 , 102.0781893 , 102.0781893 , 102.29766804,
        102.54458162, 102.66803841, 103.18930041, 103.80658436,
        104.42386831, 105.13717421, 105.90534979, 107.35025149]])


def _bright_hue_avg_20_20(hue):
    J = np.interp(hue % 360, _hues, _bright_hues_avg_20_20[0])
    C = np.interp(hue % 360, _hues, _bright_hues_avg_20_20[1]) - 0.1
    if hue > 359: C -= .4
    return J, C

def _bright_hq_avg_20_20(hq):
    J = np.interp(hq % 400, _hqs, _bright_hqs_avg_20_20[0])
    C = np.interp(hq % 400, _hqs, _bright_hqs_avg_20_20[1]) - 0.1
    return J, C

@lru_cache(maxsize=16384)
def _find_bright(cam16, descr, clr, idx=1, guess=10., step=10., tol=1e-1):
    clr = list(clr)
    digg = True
    def cfunc(c):
        clr[idx] = c
        return _srgb_linear.from_xyz100(cam16.to_xyz100(clr, descr))
    for i in range(300):
        colors = cfunc(guess)
        is_clipped = np.any((colors > 1) + (colors < 0))
        if not is_clipped:
            digg = False
        if not digg and is_clipped:
            guess -= step
            step /= 3.
        elif step < tol:
            return guess
        guess += step

def _make_name_search_tree():
    cam16ucs = CAM16UCS(0.69, 20, 20, True, whitepoints_cie1931['D65'])
    names, colors = zip(*COLORS.items())
    named_srgb = np.array(colors)
    named_xyz = _srgb_linear.to_xyz100(_srgb_linear.from_srgb1(named_srgb.T))
    named_cam16ucs = cam16ucs.from_xyz100(named_xyz).T
    return KDTree(named_cam16ucs), names

_named_kdt, _named_colors = _make_name_search_tree()

@lru_cache(maxsize=4096)
def _search_nearest_name(*cam16ucs_color):
    return _named_colors[_named_kdt.query(cam16ucs_color)[1]]

def _srgb_to_linear(srgb):
    a = 0.055
    return tuple(
        c / 12.92 if c <= 0.040449936 else ((c + a) / (1 + a)) ** 2.4
        for c in srgb)

def _srgb_from_linear(srgb_linear):
    a = 0.055
    return tuple(
        c * 12.92 if c <= 0.0031308 else (1 + a) * c ** (1 / 2.4) - a
        for c in srgb_linear)

# def _srgba_bytes_to_linear(pixels):
#     pixels_rgba = np.frombuffer(pixels, dtype='ubyte').reshape(len(pixels)//4, 4)
#     pixels_linear = pixels_rgba[:, :3] // 2 #_srgb_linear.from_srgb1(pixels_rgba[:, :3]).astype('ubyte')
#     return np.concatenate([pixels_linear, pixels_rgba[:, 3:]], axis=1).tobytes()

    # 'J': (0, 0, 'lightness'),
    # 'Q': (6, 0, 'brightness'),
    # 'C': (1, 1, 'chroma'),
    # 'M': (4, 1, 'colorfulness'),
    # 's': (5, 1, 'saturation'),
    # 'H': (2, 2, 'hue quadrature'),
    # 'h': (3, 2, 'hue')



def _parse_srgb(*color):
    if len(color) == 1 and isinstance(color[0], (tuple, list)):
        color = color[0]

    clr_len = len(color)
    alpha = None

    if clr_len == 1 and isinstance(color[0], str):
        color = color[0]
        clr_len = len(color)
        if color.startswith('#'):
            if clr_len in (7, 9):
                if clr_len == 9:
                    alpha = int(color[7:9], 16) / 255
                return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)), alpha
            elif clr_len in (4, 5):
                clr = tuple(16 * int(h, 16) / 255 for h in color[1:])
                if clr_len == 5:
                    alpha = clr[3]
                return clr[:3], alpha
        elif color in COLORS:
            return COLORS[color], None
    elif clr_len in (3, 4):
        clr = tuple(map(float, color))
        if clr_len == 4:
            alpha = clr[3]
        return clr[:3], alpha
    raise AttributeError("cannot read color: %s" % repr(color))

# @lru_cache(maxsize=16384)
def _parse_color(cam16ucs, *largs, mode=None, opacity=100.0, **kwargs):
    xyz = None
    color = None
    srgb_color = None
    linear_srgb_color = None
    is_clipped = None
    _mode = list('Jsh')
    _opacity = opacity
    cam16 = cam16ucs.cam16
    if mode == 'sRGB' or (largs and mode is None):
        clr, alpha = _parse_srgb(*largs)
        is_clipped = any([c < 0. or c > 1. for c in clr])
        srgb_color = [max(min(c, 1.), 0.) for c in clr]
        linear_srgb_color = _srgb_to_linear(srgb_color)
        if alpha is not None:
            _opacity *= alpha
    elif not largs and mode is None:
        mode2 = [None] * 3
        clr = [0., 0., 0.]  # np.array([0., 0., 0.])
        for dim in _dims:
            ltr, _, pos = _dims[dim]
            if dim in kwargs:
                clr[pos] = kwargs[dim]
                mode2[pos] = ltr
            if ltr in kwargs:
                clr[pos] = kwargs[ltr]
                mode2[pos] = ltr
        clr[2] %= 400 if mode2[2] == 'H' else 360
        empty_args = mode2.count(None)
        if empty_args == 0:
            pass
        elif empty_args == 3:
            mode2 = 'JCh'
            clr = [50., 0., 0.] # np.array([50., 0., 0.])
        elif empty_args == 2:
            # FIXME: use _find_bright for custom _own_cam
            if mode2[2] == 'h':
                J, C = _bright_hue_avg_20_20(clr[2])
                s = 50 * sqrt(cam16.c * C * 10 / sqrt(J) / (cam16.A_w + 4))
                clr[0] = J
                clr[1] = s
                mode2 = 'Jsh'
            elif mode2[2] == 'H':
                J, C = _bright_hq_avg_20_20(clr[2])
                s = 50 * sqrt(cam16.c * C * 10 / sqrt(J) / (cam16.A_w + 4))
                clr[0] = J
                clr[1] = s
                mode2 = 'JsH'
            elif mode2[0]:
                mode2[1] = 's'
                mode2[2] = 'h'
            else:
                raise AttributeError('wrong color definition')

        elif mode2[0] is None:
            mode2[0] = 'J'
            J = _find_bright(cam16, tuple(mode2), tuple(clr), 0)
            if J is None:
                raise AttributeError('no such color exists')
            clr[0] = J
        elif mode2[1] is None:
            mode2[1] = 'C'
            C = _find_bright(cam16, tuple(mode2), tuple(clr), 1)
            if C is None:
                raise AttributeError('no such color exists')
            clr[1] = C
        else:
            raise AttributeError('wrong color definition')
        _mode = list(mode2)
        color = [clr[_ltr2ord[ltr]] if ltr in mode2 else None for ltr in _ltrs]
        # xyz = cam16.to_xyz100(clr, mode2)
    else:
        # largs = np.array(largs)
        if mode == 'sRGB8':
            alpha = None
            if len(largs) == 4:
                alpha = largs[3]
            srgb_color = [c / 255 for c in largs[:3]]
            linear_srgb_color = _srgb_to_linear(srgb_color)
            is_clipped = any([c < 0.0 or c > 1.0 for c in srgb_color])
            if alpha is not None:
                _opacity *= alpha
        elif mode == 'Linear sRGB':
            xyz = _srgb_linear.to_xyz100(np.array(largs))
        elif mode == 'CIELAB':
            xyz = _cielab.to_xyz100(np.array(largs))
        elif mode == 'CIELUV':
            xyz = CIELUV().to_xyz100(np.array(largs))
        elif mode == 'CIELCH':
            xyz = CIELCH().to_xyz100(np.array(largs))
        elif mode == 'XYZ':
            xyz = np.array(largs)
        elif mode and mode[0] in 'JQ' and mode[1] in 'CMs' and mode[2] in 'Hh':
            _mode = list(mode)
            color = [largs[_ltr2ord[ltr]] if ltr in mode else None for ltr in _ltrs]
            if mode[2] == 'H':
                color[2] %= 400
            else:
                color[3] %= 360
        elif mode == 'CAM16-UCS':
            xyz = cam16ucs.to_xyz100(largs)
        else:
            raise AttributeError('cannot parse color definition')
        # TODO: Optimize converting to self._color

    return color, xyz, linear_srgb_color, srgb_color, is_clipped, _mode, _opacity
    # return copy(color), copy(xyz), linear_srgb_color, copy(srgb_color), is_clipped, copy(_mode), _opacity
    # return deepcopy((color, xyz, linear_srgb_color, srgb_color, is_clipped, _mode, _opacity))


# def _alpha_correction_poly(cam16, degree=3, plot=False):
#     srgb = _srgb_linear
#     lightness = np.linspace(1, 100, 100)
#     grayscale_cam16 = np.array([light, np.zeros_like(lightness), np.zeros_like(lightness)])
#     grayscale_srgb = srgb.to_srgb1(srgb.from_xyz100(cam16.to_xyz100(grayscale_cam16, 'JCh')))
#     mean_values = np.mean(grayscale_srgb, axis=0)
#     c = polynomial.polyfit(lightness, mean_values, degree)
#     if plot:
#         import matplotlib.pyplot as plt
#         poly = polynomial.Polynomial(c)
#         plt.plot(lightness, mean_values, 'o', label='Data')
#         plt.plot(lightness, poly(lightness), label='Fit')
#         plt.plot(lightness, 100 * (mean_values - poly(lightness)), label='Difference')
#         plt.show()
#     return c


# class _Cam16ucsSpaces(defaultdict):
#     def __missing__(self, key):
#         self[key] = value = CAM16UCS(*key)
#         return value


# class _AlphaCorrectionPolynomials(defaultdict):
#     def __missing__(self, cam16):
#         self[key] = value = _alpha_correction_poly(cam16)
#         return value


_cam16_viewing_conditions = dict(average=0.69, dim=0.59, dark=0.525)

def _cam16_specification(**kwargs):
    c = float(kwargs.get('c') or \
        _cam16_viewing_conditions[kwargs.get('surround') or 'average'])
    Y_b = float(kwargs.get('Y_b') or kwargs.get('background_luminance') or 20)
    exact_inversion = bool(kwargs.get('exact_inversion')) or True
    whitepoint = kwargs.get('whitepoint', 'D65')
    if isinstance(whitepoint, str):
        whitepoint = whitepoints_cie1931[whitepoint]
    L_A = kwargs.get('L_A') or kwargs.get('adapting_field_luminance') or \
        kwargs.get('L_W', kwargs.get('screen_luminance') or 100) * Y_b / whitepoint[1]
    return c, Y_b, L_A, exact_inversion, whitepoint


class Color(KeepWeakRefs):
    __slots__ = (
        '_srgb_color', '_linear_srgb_color', '_xyz_color', '_color',
        '_mode', '_opacity', '_own_cam', '_own_cam_spec', '_modified',
        '_is_clipped', '_show_clipped')

    # _cam16ucs_spaces = _Cam16ucsSpaces()
    _cam16_spec = (0.69, 20, 20, True, whitepoints_cie1931['D65'])
    _cam16ucs = CAM16UCS(*_cam16_spec)

    # lightness=None, brightness=None, chroma=None, colorfulness=None,
    # saturation=None, hue_quadrature=None, hue=None,
    def __init__(self, *largs, mode=None, opacity=100., clipped='show', **kwargs):
        """Make a new Color.

        :param str mode: can be 'sRGB', 'sRGB8', 'Linear sRGB', 'CIELAB',
            'CIELUV', 'CIELCH', 'XYZ', 'CAM16-UCS' or CAM16 description:
            J or Q
        """
        self._own_cam_spec = Color._cam16_spec
        self._own_cam = Color._cam16ucs
        if kwargs:
            cam16_spec = _cam16_specification(**kwargs)
            if cam16_spec != Color._cam16_spec:
                self._own_cam_spec = cam16_spec
                self._own_cam = CAM16UCS(*cam16_spec)

        res = _parse_color(self._own_cam, *largs, mode=mode, opacity=opacity, **kwargs)
        self._color, self._xyz_color, self._linear_srgb_color, self._srgb_color, \
        self._is_clipped, self._mode, self._opacity = res #(copy(r) for r in res)
        self._show_clipped = clipped == 'show' or clipped is True
        # self._color = res[0].copy() if res[0] else None
        # self._xyz_color = res[1].copy() if res[1] else None
        # self._linear_srgb_color = res[2]
        # self._srgb_color = copy(res[3])
        # self._is_clipped = res[4]
        # self._mode = copy(res[5])
        # self._opacity = res[6]
        super(Color, self).__init__()

        # self._original_colors = None
        # self._dimensions = 'Jsh'

    # def __iter__(self):
    #     for c in self.srgb:
    #         yield c

    # def __len__(self):
    #     return 3

    def __copy__(self):
        clr = Color.__new__(Color)
        clr._color = copy(self._color)
        clr._xyz_color = copy(self._xyz_color)
        clr._linear_srgb_color = self._linear_srgb_color
        clr._srgb_color = copy(self._srgb_color)
        clr._is_clipped = self._is_clipped
        clr._mode = copy(self._mode)
        clr._opacity = self._opacity
        clr._show_clipped = self._show_clipped
        if self._own_cam_spec != Color._cam16_spec:
            clr._own_cam_spec = copy(self._own_cam_spec)
            clr._own_cam = CAM16UCS(*clr._own_cam_spec)
        else:
            clr._own_cam_spec = Color._cam16_spec
            clr._own_cam = Color._cam16ucs
        super(Color, clr).__init__()
        return clr

    copy = __copy__

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, Color):
            other = Color(*other)
        if self._color and other._color and self._color == other._color:
            return True
        if self._own_cam_spec == other._own_cam_spec:
            if self._srgb_color and other._srgb_color \
                    and self._srgb_color == other._srgb_color:
                return True
            if self._xyz_color and other._xyz_color \
                    and self._xyz_color == other._xyz_color:
                return True
        if not (self._color and self._color[0] and self._color[1] \
                and self._color[3]):
            self._update_cam16()
        if not (other._color and other._color[0] and other._color[1] \
                and other._color[3]):
            other._update_cam16()
        return self._color[:2] == other._color[:2] and \
            self._color[3] == other._color[3]

    def __hash__(self):
        if not (self._color and self._color[0] and (self._color[1] or self._color[5]) \
                and self._color[3]):
            self._update_cam16()
        elif self._color[1] is None:
            cam16 = self._own_cam.cam16
            J = self._color[0]
            s = self._color[5]
            C = self._color[1] = s**2 * sqrt(J) * (cam16.A_w + 4) / 25000 / cam16.c
        return hash((*self._color[:2], self._color[3]))

    def __repr__(self):
        name = None
        if self._srgb_color is not None:
            name = INV_COLORS.get(tuple(self._srgb_color))
        return f"Color('{name or self.hex}')"

    def _update_cam16(self):
        # print('_update_cam16')
        # f = inspect.currentframe()
        # tb = []
        # depth = 0
        # while f is not None:
        #     filename = f.f_code.co_filename
        #     lineno = f.f_lineno
        #     if depth > 5 and filename != '<code-input>':
        #         break
        #     print('depth', depth, filename, lineno)
        #     depth += 1
        #     f = f.f_back
        if self._xyz_color is None:
            if self._color:
                self._update_from_cam16()
            else:
                srgb = _srgb_linear
                self._xyz_color = srgb.to_xyz100(self._linear_srgb_color)  # srgb.from_srgb1(self._srgb_color))
        self._color = list(self._own_cam.cam16.from_xyz100(self._xyz_color))

    def _update_from_cam16(self):
        data = [self._color[_ltr2pos[c]] for c in self._mode]
        cam16 = self._own_cam.cam16
        self._invalidate()
        self._xyz_color = cam16.to_xyz100(data, self._mode)
        # self._color = cam16.from_xyz100(xyz)  # Update other variables
        # print('_update_from_cam16')
        # f = inspect.currentframe()

    def _invalidate(self):
        for ltr in _ltrs:
            if ltr not in self._mode:
                self._color[_ltr2pos[ltr]] = None
        self._is_clipped = None
        self._srgb_color = None
        self._linear_srgb_color = None
        self._xyz_color = None
        self._modified = True

    def _update_srgb_from_xyz(self):
        linear_srgb = _srgb_linear.from_xyz100(self._xyz_color)
        self._is_clipped = False
        if np.any(np.isnan(linear_srgb)):
            linear_srgb = np.nan_to_num(linear_srgb, False)
            self._is_clipped = True
        if np.any((linear_srgb < 0.) + (linear_srgb > 1.)):
            self._is_clipped = True
            linear_srgb.clip(0., 1., linear_srgb)
        self._linear_srgb_color = linear_srgb
        self._srgb_color = _srgb_from_linear(linear_srgb)

    @property
    def srgb(self):
        if self._srgb_color is None:
            if self._xyz_color is None:
                self._update_from_cam16()
            self._update_srgb_from_xyz()
        return tuple(self._srgb_color)

    @property
    def srgba(self):
        alpha = self.opacity / 100
        return self.srgb + (alpha,)

    @srgb.setter
    def srgb(self, value):
        clr, alpha = _parse_srgb(value)
        self._srgb_color = clr
        if alpha is not None:
            self.opacity = alpha * 100
        self._is_clipped = any([c < 0.0 or c > 1.0 for c in self._srgb_color])
        self._linear_srgb_color = _srgb_to_linear(clr)
        self._xyz_color = None
        self._color = None
        self._modified = True

    @property
    def linear_srgb(self):
        if self._linear_srgb_color is None:
            if self._xyz_color is None:
                self._update_from_cam16()
            self._update_srgb_from_xyz()
        return tuple(self._linear_srgb_color)

    @linear_srgb.setter
    def linear_srgb(self, value):
        clr = tuple(map(float, value))
        if len(clr) == 4:
            self.opacity = clr[3] * 100
            clr = clr[:3]
        self._linear_srgb_color = clr
        self._srgb_color = _srgb_from_linear(clr)
        self._is_clipped = any([c < 0.0 or c > 1.0 for c in self._srgb_color])
        self._xyz_color = None
        self._color = None
        self._modified = True

    @property
    def linear_srgba(self):
        alpha = self.opacity / 100
        return self.linear_srgb + (alpha,)

    @property
    def is_clipped(self):
        if self._is_clipped is None:
            self.srgb
        return self._is_clipped

    @property
    def hex(self):
        return f'#{"".join(["%02x" % int(c * 255 + 0.5) for c in self.srgb])}'

    @property
    def nearest_name(self):
        return _search_nearest_name(*self._own_cam.from_xyz100(self.xyz))

    @property
    def xyz(self):
        if self._xyz_color is None:
            srgb = _srgb_linear
            if self._linear_srgb_color is not None:
                self._xyz_color = srgb.to_xyz100(self._linear_srgb_color) # srgb.from_srgb1(self._srgb_color))
            else:
                data = [self._color[_ltr2pos[c]] for c in self._mode]
                cam16 = self._own_cam.cam16
                self._xyz_color = cam16.to_xyz100(data, self._mode)
            # srgb_color = srgb.to_srgb1(srgb.from_xyz100(self._xyz_color))
            # self._srgb_color = srgb_color
        return self._xyz_color

    @xyz.setter
    def xyz(self, value):
        self._xyz_color = tuple(map(float, value))
        self._srgb_color = None
        self._linear_srgb_color = None
        self._is_clipped = None
        self._color = None
        self._modified = True

    @property
    def lightness(self):
        if self._color is None:
            self._update_cam16()
        return self._color[0]

    @lightness.setter
    def lightness(self, value):
        if self._color is None or self._color[0] is None:
            self._update_cam16()
        if self._color[0] != value:
            self._color[0] = value
            self._mode[0] = 'J'
            self._invalidate()

    @property
    def chroma(self):
        if self._color is None or self._color[1] is None and \
                (self._color[0] is None or self._color[5] is None):
            self._update_cam16()
        elif self._color[1] is None:
            cam16 = self._own_cam.cam16
            J = self._color[0]
            s = self._color[5]
            C = self._color[1] = s**2 * sqrt(J) * (cam16.A_w + 4) / 25000 / cam16.c
        return self._color[1]

    @chroma.setter
    def chroma(self, value):
        if self._color is None or self._color[1] is None:
            self._update_cam16()
        if self._color[1] != value:
            self._color[1] = value
            self._mode[1] = 'C'
            self._invalidate()

    @property
    def hue_quadrature(self):
        if self._color is None: # or self._color[2] is None:
            self._update_cam16()
        elif self._color[2] is None:
            cam16 = self._own_cam.cam16
            h = self._color[3]
            h_ = (h - cam16.h[0]) % 360 + cam16.h[0]
            e_t = (cos(radians(h_) + 2) + 3.8) / 4
            i = np.searchsorted(cam16.h, h_) - 1
            beta = (h_ - cam16.h[i]) * cam16.e[i + 1]
            H = self._color[2] = cam16.H[i] + 100 * beta / (beta + \
                cam16.e[i] * (cam16.h[i + 1] - h_))
        return self._color[2]

    @hue_quadrature.setter
    def hue_quadrature(self, value):
        if self._color is None or self._color[2] is None:
            self._update_cam16()
        if self._color[2] != value % 400:
            self._color[2] = value % 400
            self._mode[2] = 'H'
            self._invalidate()

    @property
    def hue(self):
        if self._color is None:
            self._update_cam16()
        elif self._color[3] is None:
            cam16 = self._own_cam.cam16
            H = self._color[2]
            i = np.searchsorted(cam16.H, H) - 1
            Hi = cam16.H[i]
            hi, hi1 = cam16.h[i], cam16.h[i + 1]
            ei, ei1 = cam16.e[i], cam16.e[i + 1]
            h = self._color[3] = ((H - Hi) * (ei1 * hi - ei * hi1) - 100 * hi * ei1) / (
                (H - Hi) * (ei1 - ei) - 100 * ei1
            ) % 360
        return self._color[3]

    @hue.setter
    def hue(self, value):
        if self._color is None or self._color[3] is None:
            self._update_cam16()
        if self._color[3] != value % 360:
            self._color[3] = value % 360
            self._mode[2] = 'h'
            self._invalidate()

    @property
    def colorfulness(self):
        if self._color is None or self._color[4] is None:
            self._update_cam16()
        return self._color[4]

    @colorfulness.setter
    def colorfulness(self, value):
        if self._color is None or self._color[4] is None:
            self._update_cam16()
        if self._color[4] != value:
            self._color[4] = value
            self._mode[1] = 'M'
            self._invalidate()

    @property
    def saturation(self):
        if self._color is None or self._color[5] is None and (self._color[0] is None or self._color[1] is None):
            self._update_cam16()
        elif self._color[5] is None:
            cam16 = self._own_cam.cam16
            J = self._color[0]
            C = self._color[1]
            s = self._color[5] = 50 * sqrt(cam16.c * C * 10 /sqrt(J) / (cam16.A_w + 4))
        return self._color[5]

    @saturation.setter
    def saturation(self, value):
        if self._color is None or self._color[5] is None:
            self._update_cam16()
        if self._color[5] != value:
            self._color[5] = value
            self._mode[1] = 's'
            self._invalidate()

    @property
    def brightness(self):
        if self._color is None or self._color[6] is None:
            self._update_cam16()
        return self._color[6]

    @brightness.setter
    def brightness(self, value):
        if self._color is None or self._color[6] is None:
            self._update_cam16()
        if self._color[6] != value:
            self._color[6] = value
            self._mode[0] = 'Q'
            self._invalidate()

    @property
    def opacity(self):
        if self._is_clipped and not self._show_clipped:
            return 0.
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        self._opacity = max(0, min(value, 100))
        self._modified = True


def _global_update_colors():
    need_xyz = defaultdict(list)
    need_srgb = []
    for clr in Color.get_instances():
        if clr._linear_srgb_color is None:
            if clr._xyz_color is None:
                need_xyz[tuple(clr._mode), clr._own_cam.cam16].append(clr)
            need_srgb.append(clr)

    for (mode, cam16), clrs in need_xyz.items():
        data = np.array([[clr._color[_ltr2pos[c]] for c in clr._mode] for clr in clrs])
        xyzs = cam16.to_xyz100(data.T, mode).T
        for clr, xyz in zip(clrs, xyzs):
            clr._xyz_color = xyz

    if need_srgb:
        data = np.array([clr._xyz_color for clr in need_srgb])
        lsrgb_clrs = _srgb_linear.from_xyz100(data.T).T
        clipped = np.any((lsrgb_clrs > 1.) + (lsrgb_clrs < 0.) + np.isnan(lsrgb_clrs), axis=1)
        lsrgb_clrs = np.nan_to_num(lsrgb_clrs, False)
        lsrgb_clrs.clip(0., 1., lsrgb_clrs)
        srgb_clrs = _srgb_linear.to_srgb1(lsrgb_clrs.T).T
        for clr, lin_srgb, srgb, clip in zip(need_srgb, lsrgb_clrs, srgb_clrs, clipped):
            clr._linear_srgb_color = lin_srgb
            clr._srgb_color = srgb
            clr._is_clipped = clip

# from numpy.polynomial import polynomial as P
# light = np.linspace(1, 100, 100)
# values = np.array([sum(Color(i, 0, 0, mode='JCh').srgb) / 3 * 100 for i in light])
# c, stats = P.polyfit(light, values, 3, full=True)
# plt.plot(x, y, 'o', label='Data')
# plt.plot(x, c[3] * x**3 + c[2] * x**2 + c[1] * x + c[0], label='Fit')
# plt.show()
