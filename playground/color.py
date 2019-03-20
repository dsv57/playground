from collections import defaultdict

import numpy as np
from numpy import sin, cos, arctan2 as atan2, \
                  sqrt, ceil, floor, degrees, radians, log, pi, exp, transpose
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

def _make_name_search_tree():
    cam16ucs = CAM16UCS(0.69, 20, 64 / pi / 5, True, whitepoints_cie1931['D65'])
    names, colors = zip(*COLORS.items())
    named_srgb = np.array(colors)
    named_xyz = _srgb_linear.to_xyz100(_srgb_linear.from_srgb1(named_srgb.T))
    named_cam16ucs = cam16ucs.from_xyz100(named_xyz).T
    return KDTree(named_cam16ucs), names

_named_kdt, _named_colors = _make_name_search_tree()

def _search_nearest_name(cam16ucs_color):
    return _named_colors[_named_kdt.query(cam16ucs_color)[1]]



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

    if len(color) in [3, 4]:
            # and all([isinstance(c, (float, int)) for c in color]):
        return tuple(map(float, color))
    elif len(color) == 1 and isinstance(color[0], str):
        color = color[0]
        if color.startswith('#'):
            if len(color) == 7:
                return [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            elif len(color) == 4:
                return [16 * int(h, 16) / 255 for h in color[1:]]
        elif color in COLORS:
            return COLORS[color]
    raise AttributeError("bad colorstring: %s" % color)


class _Cam16ucsSpaces(defaultdict):
    def __missing__(self, key):
        self[key] = value = CAM16UCS(*key)
        return value


class Color(object):
    __slots__ = ('_srgb_color', '_xyz_color', '_color', '_description', '_cam16ucs')

    _cam16ucs_spaces = _Cam16ucsSpaces()

    def __init__(self, *largs, description='sRGB', **kwargs):
        c = kwargs.get('c', 0.69)
        Y_b = kwargs.get('Y_b', 20)
        L_A = kwargs.get('L_A', 64 / pi / 5)
        exact_inversion = kwargs.get('exact_inversion', True)
        whitepoint = kwargs.get('whitepoint', 'D65')
        if isinstance(whitepoint, str):
            whitepoint = whitepoints_cie1931[whitepoint]

        cam16ucs = Color._cam16ucs_spaces[c, Y_b, L_A, exact_inversion, whitepoint]
        cam16 = cam16ucs.cam16
        self._cam16ucs = cam16ucs
        # self._cam16 = cam16
        # self._srgb = SrgbLinear()
        # TODO: alpha-channel
        # srgb = _srgb_linear
        # cielab = _cielab

        xyz = None
        srgb_color = None
        # descr = 'Jsh'
        self._color = None  # cam16.from_xyz100(xyz) if xyz else None
        self._description = list('Jsh')
        if description == 'sRGB':
            srgb_color = _from_srgb(*largs)
        elif description == 'CIELAB':
            xyz = _cielab.to_xyz100(largs)
        elif description == 'CIELUV':
            xyz = CIELUV().to_xyz100(largs)
        elif description == 'CIELCH':
            xyz = CIELCH().to_xyz100(largs)
        elif description == 'XYZ':
            xyz = largs
        elif description[0] in 'JQ' and description[1] in 'CMs' and description[2] in 'Hh':
            xyz = cam16.to_xyz100(largs, description)
            self._description = list(description)
            self._color = [largs[_ltr2ord[ltr]] if ltr in description else None for ltr in _ltrs]
        elif description in ['CAM16UCS', 'CAM16-UCS']:
            xyz = cam16ucs.to_xyz100(largs)
            # TODO: Optimize converting to self._color

        self._xyz_color = xyz
        self._srgb_color = srgb_color
        # self._original_colors = None
        # self._dimensions = 'Jsh'

    def __iter__(self):
        for c in self.srgb:
            yield c

    def __len__(self):
        return 3

    def _update_cam16(self):
        if self._xyz_color is None:
            srgb = _srgb_linear
            self._xyz_color = srgb.to_xyz100(srgb.from_srgb1(self._srgb_color))
        self._color = list(self._cam16ucs.cam16.from_xyz100(self._xyz_color))

    def _update_from_cam16(self):
        data = [self._color[_ltr2pos[c]] for c in self._description]
        cam16 = self._cam16ucs.cam16
        self._xyz_color = cam16.to_xyz100(data, self._description)
        # self._color = cam16.from_xyz100(xyz)  # Update other variables
        for ltr in _ltrs:
            if ltr not in self._description:
                self._color[_ltr2pos[ltr]] = None
        self._srgb_color = None

    @property
    def srgb(self):
        if self._srgb_color is None:
            if self._xyz_color is None:
                self._update_from_cam16()
            srgb = _srgb_linear
            srgb_color = srgb.to_srgb1(srgb.from_xyz100(self._xyz_color))
            self._srgb_color = srgb_color
        # np.clip(srgb_color, 0.0, 1.0, srgb_color)
        return tuple(np.clip(self._srgb_color, 0.0, 1.0))

    @srgb.setter
    def srgb(self, value):
        self._srgb_color = _from_srgb(value)
        self._xyz_color = None
        self._color = None

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
        return _search_nearest_name(self._cam16ucs.from_xyz100(self.xyz))

    @property
    def xyz(self):
        if self._xyz_color is None:
            srgb = _srgb_linear
            if self._srgb_color is not None:
                self._xyz_color = srgb.to_xyz100(srgb.from_srgb1(self._srgb_color))
            else:
                cam16 = self._cam16ucs.cam16
                self._xyz_color = cam16.to_xyz100(self._color, self._description)
            srgb_color = srgb.to_srgb1(srgb.from_xyz100(self._xyz_color))
            self._srgb_color = srgb_color
        return self._xyz_color

    @xyz.setter
    def xyz(self, value):
        self._xyz_color = tuple(map(float, value))
        self._srgb_color = None
        self._color = None

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
            self._description[0] = 'J'
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
            self._description[1] = 'C'
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
            self._description[2] = 'H'
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
            self._description[2] = 'h'
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
            self._description[1] = 'M'
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
            self._description[1] = 's'
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
            self._description[0] = 'Q'
            self._update_from_cam16()


