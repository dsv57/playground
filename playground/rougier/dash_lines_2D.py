#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (C) 2013 Nicolas P. Rougier. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY NICOLAS P. ROUGIER ''AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL NICOLAS P. ROUGIER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Nicolas P. Rougier.
# -----------------------------------------------------------------------------
import os
import numpy as np
# import OpenGL.GL as gl
# from shader import Shader

from kivy.graphics import Mesh, BindTexture, RenderContext
from kivy.resources import resource_find

from .dash_atlas import DashAtlas
from .collection import Collection


# -----------------------------------------------------------------------------
class DashLines(Collection):

    join = { 'miter' : 0,
             'round' : 1,
             'bevel' : 2 }

    caps = { ''             : 0,
             'none'         : 0,
             '.'            : 0,
             'round'        : 1,
             ')'            : 1,
             '('            : 1,
             'o'            : 1,
             'triangle in'  : 2,
             '<'            : 2,
             'triangle out' : 3,
             '>'            : 3,
             'square'       : 4,
             '['            : 4,
             ']'            : 4,
             '='            : 4,
             'butt'         : 5,
             '|'            : 5 }


    # ---------------------------------
    def __init__(self, dash_atlas = None):
        vtype = np.dtype( [('a_position', 'f4', 2),
                           ('a_segment',  'f4', 2),
                           ('a_angles',   'f4', 2),
                           ('a_tangents', 'f4', 4),
                           ('a_texcoord', 'f4', 2) ])
        utype = np.dtype( [('color',      'f4', 4),
                           ('translate',  'f4', 2),
                           ('scale',      'f4', 1),
                           ('rotate',     'f4', 1),
                           ('linewidth',  'f4', 1),
                           ('antialias',  'f4', 1),
                           ('linecaps',   'f4', 2),
                           ('linejoin',   'f4', 1),
                           ('miter_limit','f4', 1),
                           ('length',     'f4', 1),
                           ('dash_phase', 'f4', 1),
                           ('dash_period','f4', 1),
                           ('dash_index', 'f4', 1),
                           ('dash_caps',  'f4', 2),
                           ('closed',     'f4', 1)] )
        Collection.__init__(self, vtype, utype)
        # shaders = os.path.join(os.path.dirname(__file__), 'shaders')
        # vertex_shader= os.path.join( shaders, 'dash-lines-2D.vert')
        # fragment_shader= os.path.join( shaders, 'dash-lines-2D.frag')
        #fragment_shader= os.path.join( shaders, 'test.frag')
        if dash_atlas is None:
            self.dash_atlas = DashAtlas()
        else:
            self.dash_atlas = dash_atlas
        # self.shader = Shader( open(vertex_shader).read(),
                              # open(fragment_shader).read() )
        self.context = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        self.context.shader.source = resource_find('dash-lines-2D.glsl')


    # ---------------------------------
    def append( self, vertices, closed=False, color = (0,0,0,1),
                linewidth = 1.0, antialias = 1.0, linejoin = 'round', miter_limit = 10.0,
                translate = (0,0), scale = 1.0, rotate = 0.0, linecaps = ('round','round'),
                dash_pattern='dashed', dash_phase = 0.0, dash_caps = ('round','round') ):

        V,I,length = self.bake( vertices, closed=closed )
        U = np.zeros(1, self.utype)

        if closed:
            U['closed']  = 1.0
        U['linewidth']   = linewidth
        U['antialias']   = antialias
        U['color']       = color
        U['translate']   = translate
        U['scale']       = scale
        U['rotate']      = rotate
        U['linejoin']    = self.join.get(linejoin, 'round')
        U['linecaps']    = ( self.caps.get(linecaps[0], 'round'),
                             self.caps.get(linecaps[1], 'round') )
        U['miter_limit'] = miter_limit
        U['length']      = length
        if self.dash_atlas:
            dash_index, dash_period = self.dash_atlas[dash_pattern]
            print('dash_index, dash_period', dash_index, dash_period)
            U['dash_phase']  = dash_phase
            U['dash_index']  = dash_index
            U['dash_period'] = dash_period
            U['dash_caps']   = ( self.caps.get(dash_caps[0], 'round'),
                                 self.caps.get(dash_caps[1], 'round') )
        Collection.append(self,V,I,U)


    # ---------------------------------
    def bake(self, vertices, closed=False):
        """
        Bake a list of 2D vertices for rendering them as thick line. Each line
        segment must have its own vertices because of antialias (this means no
        vertex sharing between two adjacent line segments).
        """

        n = len(vertices)
        P = np.array(vertices).reshape(n,2).astype(float)
        idx = np.arange(n)  # used to eventually tile the color array

        dx,dy = P[0] - P[-1]
        d = np.sqrt(dx*dx+dy*dy)

        # If closed, make sure first vertex = last vertex (+/- epsilon=1e-10)
        if closed and d > 1e-10:
            P = np.append(P, P[0]).reshape(n+1,2)
            idx = np.append(idx, idx[-1])
            n+= 1

        V = np.zeros( len(P), dtype = self.vtype )
        V['a_position'] = P

        # Tangents & norms
        T = P[1:] - P[:-1]
        
        N = np.sqrt(T[:,0]**2 + T[:,1]**2)
        # T /= N.reshape(len(T),1)
        V['a_tangents'][+1:, :2] = T
        V['a_tangents'][0, :2] = T[-1] if closed else T[0]
        V['a_tangents'][:-1, 2:] = T
        V['a_tangents'][-1, 2:] = T[0] if closed else T[-1]

        # Angles
        T1 = V['a_tangents'][:,:2]
        T2 = V['a_tangents'][:,2:]
        A = np.arctan2( T1[:,0]*T2[:,1]-T1[:,1]*T2[:,0],
                        T1[:,0]*T2[:,0]+T1[:,1]*T2[:,1])
        V['a_angles'][:-1,0] = A[:-1] 
        V['a_angles'][:-1,1] = A[+1:]

        # Segment
        L = np.cumsum(N)
        V['a_segment'][+1:,0] = L
        V['a_segment'][:-1,1] = L
        #V['a_lengths'][:,2] = L[-1]

        # Step 1: A -- B -- C  =>  A -- B, B' -- C
        V = np.repeat(V,2,axis=0)[+1:-1]
        V['a_segment'][1:] = V['a_segment'][:-1] 
        V['a_angles'][1:] = V['a_angles'][:-1] 
        V['a_texcoord'][0::2] = -1
        V['a_texcoord'][1::2] = +1
        idx = np.repeat(idx, 2)[1:-1]

        # Step 2: A -- B, B' -- C  -> A0/A1 -- B0/B1, B'0/B'1 -- C0/C1
        V = np.repeat(V,2,axis=0)
        V['a_texcoord'][0::2,1] = -1
        V['a_texcoord'][1::2,1] = +1
        idx = np.repeat(idx, 2)

        idxs = np.resize(np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32),
                         (n-1)*(2*3))
        idxs += np.repeat(4*np.arange(n-1, dtype=np.uint32), 6)

        # I = np.resize( np.array([0,1,2,1,2,3], dtype=np.uint16), (n-1)*(2*3))
        # I += np.repeat( 4*np.arange(n-1, dtype=np.uint16), 6)

        print('L[-1]', L[-1])
        # print('I', I)
        # print('idxs', idxs)
        return V, idxs, L[-1]

    # ---------------------------------
    def set_uniforms(self, uniforms=None):
        # FIXME: use context.update?
        self.context['u_uniforms'] = 2
        self.context['u_dash_atlas'] = 3
        self.context['u_uniforms_shape'] = list(map(float, self._ushape))
        if uniforms:
            for u, v in uniforms.items():
                self.context[u] = v

    def bind_textures(self):
        if self._dirty:
            self.upload()
        # self.dash_atlas.upload()
        print('Vertices:', self.vertices.tolist())
        with self.context:
            BindTexture(texture=self._texture, index=2)
            # BindTexture(texture=self.dash_atlas.texture, index=3)

    @property
    def vertices(self):
        return self._vbuffer.vertices_data
    
    # ---------------------------------
    @property
    def indices(self):
        return self._vbuffer.indices_data

    # ---------------------------------
    def draw(self, mode='triangles'): # gl.GL_TRIANGLES, uniforms = {}):
        # NOT USED!
        with self.context:
            if self._dirty:
                self.upload()
            # shader = self.shader
            # shader.bind()
            # gl.glActiveTexture( gl.GL_TEXTURE0 )
            # shader.uniformi( 'u_uniforms', 0 )
            # gl.glBindTexture( gl.GL_TEXTURE_2D, self._ubuffer_id )
            if self.dash_atlas:
                BindTexture(texture=self.dash_atlas.texture, index=3)
            #     gl.glActiveTexture( gl.GL_TEXTURE1 )
            #     shader.uniformi('u_dash_atlas', 1)
            #     gl.glBindTexture( gl.GL_TEXTURE_2D, self.dash_atlas.texture_id )
            # for name,value in uniforms.items():
            #     # print('uniform:', name, ':', value)
            #     shader.uniform(name, value)
            # shader.uniformf('u_uniforms_shape', *self._ushape)
            # print('u_uniforms_shape', self._ushape)
            BindTexture(texture=self._texture, index=2)
            mesh = Mesh(
                fmt=self._vbuffer.vfmt,
                mode=mode,
                vertices=self._vbuffer.vertices_data,
                indices=self._vbuffer.indices_data)
            return mesh
            # mesh = self._vbuffer.draw(mode)
            # print('DL2D:', self._texture, self._vbuffer._vertices.data)
            # print('draw_mode', mode)
            # shader.unbind()
