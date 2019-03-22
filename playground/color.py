from collections import defaultdict

import numpy as np
from numpy import sin, cos, arctan2 as atan2, \
                  sqrt, ceil, floor, degrees, radians, log, pi, exp, transpose
from numpy.polynomial import polynomial
from scipy.spatial import KDTree
from colorio import CIELAB, CAM16UCS, CAM16, JzAzBz, SrgbLinear
from colorio.illuminants import whitepoints_cie1931

from named_colors import COLORS


_dims = {
    'lightness': ('J', 0, 0),
    'brightness': ('Q', 6, 0),
    'chroma': ('C', 1, 1),
    'colorfulness': ('M', 4, 1),
    'saturation': ('s', 5, 1),
    'hue quadrature': ('H', 2, 2),
    'hue': ('h', 3, 2),
}
_ltrs = list('JQCMsHh')
_ltr2pos = {ltr: pos for ltr, pos, ord in _dims.values()}
_ltr2ord = {ltr: ord for ltr, pos, ord in _dims.values()}

_srgb_linear = SrgbLinear()
_cielab = CIELAB()

INV_COLORS = {v: k for k, v in COLORS.items()}

def _make_name_search_tree():
    cam16ucs = CAM16UCS(0.69, 20, 20, True, whitepoints_cie1931['D65'])
    names, colors = zip(*COLORS.items())
    named_srgb = np.array(colors)
    named_xyz = _srgb_linear.to_xyz100(_srgb_linear.from_srgb1(named_srgb.T))
    named_cam16ucs = cam16ucs.from_xyz100(named_xyz).T
    return KDTree(named_cam16ucs), names

_named_kdt, _named_colors = _make_name_search_tree()

def _search_nearest_name(cam16ucs_color):
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


    # 'J': (0, 0, 'lightness'),
    # 'Q': (6, 0, 'brightness'),
    # 'C': (1, 1, 'chroma'),
    # 'M': (4, 1, 'colorfulness'),
    # 's': (5, 1, 'saturation'),
    # 'H': (2, 2, 'hue quadrature'),
    # 'h': (3, 2, 'hue')



def _from_srgb(*color):
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


class Color(object):
    __slots__ = (
        '_srgb_color', '_linear_srgb_color', '_xyz_color', '_color',
        '_mode', '_opacity', '_own_cam', '_own_cam_spec', '_modified')

    # _cam16ucs_spaces = _Cam16ucsSpaces()
    _cam16_spec = (0.69, 20, 20, True, whitepoints_cie1931['D65'])
    _cam16ucs = CAM16UCS(*_cam16_spec)

    def __init__(self, *largs, mode='sRGB', opacity=None, **kwargs):
        self._own_cam_spec = Color._cam16_spec
        self._own_cam = Color._cam16ucs
        if kwargs:
            cam16_spec = _cam16_specification(**kwargs)
            if cam16_spec != Color._cam16_spec:
                self._own_cam_spec = cam16_spec
                self._own_cam = CAM16UCS(*cam16_spec)
        cam16 = self._own_cam.cam16

        xyz = None
        srgb_color = None
        self._color = None
        self._srgb_color = None
        self._linear_srgb_color = None
        self._mode = list('Jsh')
        self._opacity = kwargs.get('opacity')
        if mode == 'sRGB':
            clr, alpha = _from_srgb(*largs)
            self._srgb_color = clr
            self._linear_srgb_color = _srgb_to_linear(clr)
            if self._opacity is not None and alpha is not None:
                self._opacity *= alpha
            else:
                self._opacity = alpha
        elif mode == 'Linear sRGB':
            xyz = _srgb_linear.to_xyz100(largs)
        elif mode == 'CIELAB':
            xyz = _cielab.to_xyz100(largs)
        elif mode == 'CIELUV':
            xyz = CIELUV().to_xyz100(largs)
        elif mode == 'CIELCH':
            xyz = CIELCH().to_xyz100(largs)
        elif mode == 'XYZ':
            xyz = largs
        elif mode[0] in 'JQ' and mode[1] in 'CMs' and mode[2] in 'Hh':
            xyz = cam16.to_xyz100(largs, mode)
            self._mode = list(mode)
            self._color = [largs[_ltr2ord[ltr]] if ltr in mode else None for ltr in _ltrs]
        elif mode in ['CAM16UCS', 'CAM16-UCS']:
            xyz = cam16ucs.to_xyz100(largs)
            # TODO: Optimize converting to self._color

        self._xyz_color = xyz
        # self._original_colors = None
        # self._dimensions = 'Jsh'

    def __iter__(self):
        for c in self.srgb:
            yield c

    def __len__(self):
        return 3

    def __repr__(self):
        name = None
        if self._srgb_color is not None:
            name = INV_COLORS.get(tuple(self._srgb_color))
        return f"Color('{name or self.hex}')"

    def _update_cam16(self):
        if self._xyz_color is None:
            srgb = _srgb_linear
            self._xyz_color = srgb.to_xyz100(srgb.from_srgb1(self._srgb_color))
        self._color = list(self._own_cam.cam16.from_xyz100(self._xyz_color))

    def _update_from_cam16(self):
        data = [self._color[_ltr2pos[c]] for c in self._mode]
        cam16 = self._own_cam.cam16
        self._xyz_color = cam16.to_xyz100(data, self._mode)
        # self._color = cam16.from_xyz100(xyz)  # Update other variables
        for ltr in _ltrs:
            if ltr not in self._mode:
                self._color[_ltr2pos[ltr]] = None
        self._srgb_color = None
        self._linear_srgb_color = None
        self._modified = True

    def _update_srgb_from_xyz(self):
        linear_srgb = tuple(_srgb_linear.from_xyz100(self._xyz_color))
        self._linear_srgb_color = linear_srgb
        self._srgb_color = _srgb_from_linear(linear_srgb)

    @property
    def srgb(self):
        if self._srgb_color is None:
            if self._xyz_color is None:
                self._update_from_cam16()
            self._update_srgb_from_xyz()
        return tuple(np.clip(self._srgb_color, 0.0, 1.0))

    @srgb.setter
    def srgb(self, value):
        clr, alpha = _from_srgb(value)
        self._srgb_color = clr
        if alpha is not None:
            self._opacity = alpha
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
        return tuple(np.clip(self._linear_srgb_color, 0.0, 1.0))

    @linear_srgb.setter
    def linear_srgb(self, value):
        clr = tuple(map(float, value))
        if len(clr) == 4:
            self._opacity = clr[3]
            clr = clr[:3]
        self._linear_srgb_color = clr
        self._srgb_color = _srgb_from_linear(clr)
        self._xyz_color = None
        self._color = None
        self._modified = True

    @property
    def is_clipped(self):
        if self._srgb_color is None:
            self.srgb
        return any([c < 0.0 or c > 1.0 for c in self._srgb_color])

    @property
    def hex(self):
        return f'#{"".join(["%02x" % int(c * 255 + 0.5) for c in self.srgb])}'

    @property
    def nearest_name(self):
        return _search_nearest_name(self._own_cam.from_xyz100(self.xyz))

    @property
    def xyz(self):
        if self._xyz_color is None:
            srgb = _srgb_linear
            if self._srgb_color is not None:
                self._xyz_color = srgb.to_xyz100(srgb.from_srgb1(self._srgb_color))
            else:
                cam16 = self._own_cam.cam16
                self._xyz_color = cam16.to_xyz100(self._color, self._mode)
            # srgb_color = srgb.to_srgb1(srgb.from_xyz100(self._xyz_color))
            # self._srgb_color = srgb_color
        return self._xyz_color

    @xyz.setter
    def xyz(self, value):
        self._xyz_color = tuple(map(float, value))
        self._srgb_color = None
        self._linear_srgb_color = None
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
            self._update_from_cam16()

    @property
    def chroma(self):
        if self._color is None or self._color[1] is None:
            self._update_cam16()
        return self._color[1]

    @chroma.setter
    def chroma(self, value):
        if self._color is None or self._color[1] is None:
            self._update_cam16()
        if self._color[1] != value:
            self._color[1] = value
            self._mode[1] = 'C'
            self._update_from_cam16()

    @property
    def hue_quadrature(self):
        if self._color is None or self._color[2] is None:
            self._update_cam16()
        return self._color[2]

    @hue_quadrature.setter
    def hue_quadrature(self, value):
        if self._color is None or self._color[2] is None:
            self._update_cam16()
        if self._color[2] != value % 400:
            self._color[2] = value % 400
            self._mode[2] = 'H'
            self._update_from_cam16()

    @property
    def hue(self):
        if self._color is None or self._color[3] is None:
            self._update_cam16()
        return self._color[3]

    @hue.setter
    def hue(self, value):
        if self._color is None or self._color[3] is None:
            self._update_cam16()
        if self._color[3] != value % 360:
            self._color[3] = value % 360
            self._mode[2] = 'h'
            self._update_from_cam16()

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
            self._update_from_cam16()

    @property
    def saturation(self):
        if self._color is None or self._color[5] is None:
            self._update_cam16()
        return self._color[5]

    @saturation.setter
    def saturation(self, value):
        if self._color is None or self._color[5] is None:
            self._update_cam16()
        if self._color[5] != value:
            self._color[5] = value
            self._mode[1] = 's'
            self._update_from_cam16()

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
            self._update_from_cam16()


# from numpy.polynomial import polynomial as P
# light = np.linspace(1, 100, 100)
# values = np.array([sum(Color(i, 0, 0, mode='JCh').srgb) / 3 * 100 for i in light])
# c, stats = P.polyfit(light, values, 3, full=True)
# plt.plot(x, y, 'o', label='Data')
# plt.plot(x, c[3] * x**3 + c[2] * x**2 + c[1] * x + c[0], label='Fit')
# plt.show()
