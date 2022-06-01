---VERTEX SHADER-------------------------------------------------------
#version 120
#ifdef GL_ES
    precision highp float;
#endif

// -----------------------------------------------------------------------------
// Copyright (c) 2013 Nicolas P. Rougier. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY NICOLAS P. ROUGIER ''AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL NICOLAS P. ROUGIER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are
// those of the authors and should not be interpreted as representing official
// policies, either expressed or implied, of Nicolas P. Rougier.
// -----------------------------------------------------------------------------
const float PI = 3.14159265358979323846264;
const float THETA = 15.0 * 3.14159265358979323846264/180.0;

// Cross product of v1 and v2
float cross(in vec2 v1, in vec2 v2) {
    return v1.x*v2.y - v1.y*v2.x;
}

// Returns distance of v3 to line v1-v2
float signed_distance(in vec2 v1, in vec2 v2, in vec2 v3) {
    return cross(v2-v1,v1-v3) / length(v2-v1);
}

// Rotate v around origin
void rotate( in vec2 v, in float alpha, out vec2 result ) {
    float c = cos(alpha);
    float s = sin(alpha);
    result = vec2( c*v.x - s*v.y,
                   s*v.x + c*v.y );
}

// vec2 transform_vector(vec2 x, vec2 base) {
//     vec4 o = $transform(vec4(base, 0, 1));
//     return ($transform(vec4(base+x, 0, 1)) - o).xy;
// }


// Uniforms
// ------------------------------------
uniform mat4      modelview_mat, projection_mat;
uniform sampler2D u_uniforms;
uniform vec3      u_uniforms_shape;
uniform float     u_scale;

// Attributes
// ------------------------------------
attribute vec2 a_position;
attribute vec4 a_tangents;
attribute vec2 a_segment;
attribute vec2 a_angles;
attribute vec2 a_texcoord;
attribute float a_index;

// Varying
// ------------------------------------
varying vec4  v_color;
varying vec2  v_segment;
varying vec2  v_angles;
varying vec2  v_linecaps;
varying vec2  v_texcoord;
varying vec2  v_miter;
varying float v_miter_limit;
varying float v_length;
varying float v_linejoin;
varying float v_linewidth;
varying float v_antialias;
varying float v_dash_phase;
varying float v_dash_period;
varying float v_dash_index;
varying vec2  v_dash_caps;
varying float v_closed;
varying float v_scale;

void main()
{
    // gl_Position = (projection_mat*(modelview_mat))*vec4(a_position,0.0,1.0); // my
    // return;

    // ------------------------------------------------------- Get uniforms ---
    float rows = u_uniforms_shape.x;
    float cols = u_uniforms_shape.y;
    float count= u_uniforms_shape.z;
    float index = a_index;
    int index_x = int(mod(index, (floor(cols/(count/4.0))))) * int(count/4.0);
    int index_y = int(floor(index / (floor(cols/(count/4.0)))));
    float size_x = cols - 1.0;
    float size_y = rows - 1.0;
    float ty = 0.0;
    v_scale = u_scale;
    if (size_y > 0.0)
        ty = float(index_y)/size_y;

    int i = index_x;
    vec4 _uniform;

    // Get fg_color(4)
    v_color = texture2D(u_uniforms, vec2(float(i++)/size_x,ty));
    // v_color = vec4(1.,0.,0.,1.);
    // v_color = texture2D(u_uniforms, vec2(float(i++)/size_x,ty));

    // Get translate(2), scale(1), rotate(1)
    _uniform = texture2D(u_uniforms, vec2(float(i++)/size_x,ty));
    vec2  translate = _uniform.xy;//vec2(0.,0.);//_uniform.xy;
    float scale     = _uniform.z;//1.;//_uniform.z;
    float theta     = _uniform.w;//0.;//_uniform.w;

    // Get linewidth(1), antialias(1), linecaps(2)
    _uniform = texture2D(u_uniforms, vec2(float(i++)/size_x,ty));
    v_linewidth = _uniform.x;//50.;//_uniform.x;
    v_antialias = _uniform.y;//1.;//_uniform.y;
    v_linecaps  = vec2(0.,0.);//_uniform.zw;//vec2(0.,0.);//_uniform.zw;

    // Get linejoin(1), miterlimit(1), length(1), dash_phase(1)
    _uniform = texture2D(u_uniforms, vec2(float(i++)/size_x,ty));
    v_linejoin    = 2.;//_uniform.x;//2.;//_uniform.x;
    v_miter_limit = _uniform.y;//10.;//_uniform.y;
    v_length      = _uniform.z;// BUG: ? Low value //2.*2701.326500831848; //_uniform.z; //9800.;// 3814.9
    //v_linewidth = _uniform.z / 50.; // DATA (texture uniforms) IS NOT UPDATED!!
    v_dash_phase  = _uniform.w;//0.;//_uniform.w;

    // Get dash_period(1), dash_index(1), dash_caps(2)
    _uniform = texture2D(u_uniforms, vec2(float(i++)/size_x,ty));
    v_dash_period = _uniform.x;//1.0e20;//_uniform.x;
    v_dash_index  = _uniform.y;//0.;//_uniform.y; // FIXME
    v_dash_caps   = _uniform.zw;//vec2(0.,0.);//_uniform.zw;

    // Get closed(1)
    _uniform = texture2D(u_uniforms, vec2(float(i++)/size_x,ty));
    v_closed = _uniform.x;//0.;//_uniform.x;
    bool closed = (v_closed > 0.0);
    // ------------------------------------------------------------------------


    // Attributes to varyings
    v_angles  = a_angles;
    v_segment = a_segment * scale;
    v_length  = v_length * scale;

    // Thickness below 1 pixel are represented using a 1 pixel thickness
    // and a modified alpha
    v_color.a = min(v_linewidth, v_color.a);
    v_linewidth = max(v_linewidth, 1.0);


    // If color is fully transparent we just will discard the fragment anyway
    if( v_color.a <= 0.0 )
    {
        gl_Position = vec4(0.0,0.0,0.0,1.0);
        return;
    }

    // This is the actual half width of the line
    // TODO: take care of logical - physical pixel difference here.
    float w = ceil(1.25*v_antialias+v_linewidth)/2.0;
    // float w = ceil(2.*v_antialias/u_scale+v_linewidth)/2.0;
    // float w = ceil(1.25*v_antialias+v_linewidth)/2.0;
    // float w = linewidth/2.0 + 1.5*antialias;

    vec2 position = a_position*scale;
    vec2 t1 = normalize(a_tangents.xy);
    vec2 t2 = normalize(a_tangents.zw);
    float u = a_texcoord.x;
    float v = a_texcoord.y;
    vec2 o1 = vec2( +t1.y, -t1.x);
    vec2 o2 = vec2( +t2.y, -t2.x);


    // // This is a join
    // // ----------------------------------------------------------------
    // if( t1 != t2 ) {
    //     float angle  = atan (t1.x*t2.y-t1.y*t2.x, t1.x*t2.x+t1.y*t2.y);
    //     vec2 t  = normalize(t1+t2);
    //     vec2 o  = vec2( + t.y, - t.x);

    //     if ( v_dash_index > 0.0 )
    //     {
    //         // Broken angle
    //         // ----------------------------------------------------------------
    //         if( (abs(angle) > THETA) ) {
    //             position += v * w * o / cos(angle/2.0);
    //             float s = sign(angle);
    //             if( angle < 0.0 ) {
    //                 if( u == +1.0 ) {
    //                     u = v_segment.y + v * w * tan(angle/2.0);
    //                     if( v == 1.0 ) {
    //                         position -= 2.0 * w * t1 / sin(angle);
    //                         u -= 2.0 * w / sin(angle);
    //                     }
    //                 } else {
    //                     u = v_segment.x - v * w * tan(angle/2.0);
    //                     if( v == 1.0 ) {
    //                         position += 2.0 * w * t2 / sin(angle);
    //                         u += 2.0*w / sin(angle);
    //                     }
    //                 }
    //             } else {
    //                 if( u == +1.0 ) {
    //                     u = v_segment.y + v * w * tan(angle/2.0);
    //                     if( v == -1.0 ) {
    //                         position += 2.0 * w * t1 / sin(angle);
    //                         u += 2.0 * w / sin(angle);
    //                     }
    //                 } else {
    //                     u = v_segment.x - v * w * tan(angle/2.0);
    //                     if( v == -1.0 ) {
    //                         position -= 2.0 * w * t2 / sin(angle);
    //                         u -= 2.0*w / sin(angle);
    //                     }
    //                 }
    //             }
    //             // Continuous angle
    //             // ------------------------------------------------------------
    //         } else {
    //             position += v * w * o / cos(angle/2.0);
    //             if( u == +1.0 ) u = v_segment.y;
    //             else            u = v_segment.x;
    //         }
    //     }

    //     // Solid line
    //     // --------------------------------------------------------------------
    //     else
    //     {
    //         position.xy += v * w * o / cos(angle/2.0);
    //         if( angle < 0.0 ) {
    //             if( u == +1.0 ) {
    //                 u = v_segment.y + v * w * tan(angle/2.0);
    //             } else {
    //                 u = v_segment.x - v * w * tan(angle/2.0);
    //             }
    //         } else {
    //             if( u == +1.0 ) {
    //                 u = v_segment.y + v * w * tan(angle/2.0);
    //             } else {
    //                 u = v_segment.x - v * w * tan(angle/2.0);
    //             }
    //         }
    //     }

    // // This is a line start or end (t1 == t2)
    // // ------------------------------------------------------------------------
    // } else {
    //     position += v * w * o1;
    //     if( u == -1.0 ) {
    //         u = v_segment.x - w;
    //         position -=  w * t1;
    //     } else {
    //         u = v_segment.y + w;
    //         position +=  w * t2;
    //     }
    // }

    // // Miter distance
    // // ------------------------------------------------------------------------
    // vec2 t;
    // vec2 curr = a_position*scale;
    // if( a_texcoord.x < 0.0 ) {
    //     vec2 next = curr + t2*(v_segment.y-v_segment.x);

    //     rotate( t1, +a_angles.x/2.0, t);
    //     v_miter.x = signed_distance(curr, curr+t, position);

    //     rotate( t2, +a_angles.y/2.0, t);
    //     v_miter.y = signed_distance(next, next+t, position);
    // } else {
    //     vec2 prev = curr - t1*(v_segment.y-v_segment.x);

    //     rotate( t1, -a_angles.x/2.0,t);
    //     v_miter.x = signed_distance(prev, prev+t, position);

    //     rotate( t2, -a_angles.y/2.0,t);
    //     v_miter.y = signed_distance(curr, curr+t, position);
    // }

    // if (!closed && v_segment.x <= 0.0) {
    //     v_miter.x = 1e10;
    // }
    // if (!closed && v_segment.y >= v_length)
    // {
    //     v_miter.y = 1e10;
    // }

    // v_texcoord = vec2( u, v*w );

    // // Rotation
    // float c = cos(theta);
    // float s = sin(theta);
    // position.xy = vec2( c*position.x - s*position.y,
    //                     s*position.x + c*position.y );
    // // Translation
    // position += translate;

    // This is a join
    // ----------------------------------------------------------------
    if( t1 != t2 ) {
        float angle  = atan (t1.x*t2.y-t1.y*t2.x, t1.x*t2.x+t1.y*t2.y);
        vec2 t  = normalize(t1+t2);
        vec2 o  = vec2( + t.y, - t.x);

        if ( v_dash_index > 0.0 )
        {
            // Broken angle
            // ----------------------------------------------------------------
            if( (abs(angle) > THETA) ) {
                position += v * w * o / cos(angle/2.0);
                float s = sign(angle);
                if( angle < 0.0 ) {
                    if( u == +1.0 ) {
                        u = v_segment.y + v * w * tan(angle/2.0);
                        if( v == 1.0 ) {
                            position -= 2.0 * w * t1 / sin(angle);
                            u -= 2.0 * w / sin(angle);
                        }
                    } else {
                        u = v_segment.x - v * w * tan(angle/2.0);
                        if( v == 1.0 ) {
                            position += 2.0 * w * t2 / sin(angle);
                            u += 2.0*w / sin(angle);
                        }
                    }
                } else {
                    if( u == +1.0 ) {
                        u = v_segment.y + v * w * tan(angle/2.0);
                        if( v == -1.0 ) {
                            position += 2.0 * w * t1 / sin(angle);
                            u += 2.0 * w / sin(angle);
                        }
                    } else {
                        u = v_segment.x - v * w * tan(angle/2.0);
                        if( v == -1.0 ) {
                            position -= 2.0 * w * t2 / sin(angle);
                            u -= 2.0*w / sin(angle);
                        }
                    }
                }
                // Continuous angle
                // ------------------------------------------------------------
            } else {
                position += v * w * o / cos(angle/2.0);
                if( u == +1.0 ) u = v_segment.y;
                else            u = v_segment.x;
            }
        }

        // Solid line
        // --------------------------------------------------------------------
        else
        {
            position.xy += v * w * o / cos(angle/2.0);
            if( angle < 0.0 ) {
                if( u == +1.0 ) {
                    u = v_segment.y + v * w * tan(angle/2.0);
                } else {
                    u = v_segment.x - v * w * tan(angle/2.0);
                }
            } else {
                if( u == +1.0 ) {
                    u = v_segment.y + v * w * tan(angle/2.0);
                } else {
                    u = v_segment.x - v * w * tan(angle/2.0);
                }
            }
        }

    // This is a line start or end (t1 == t2)
    // ------------------------------------------------------------------------
    } else {
        position += v * w * o1;
        if( u == -1.0 ) {
            u = v_segment.x - w;
            position -=  w * t1;
        } else {
            u = v_segment.y + w;
            position +=  w * t2;
        }
    }

    // Miter distance
    // ------------------------------------------------------------------------
    vec2 t;
    //vec2 curr = $transform(vec4(a_position,0.,1.)).xy*u_scale;
    // vec2 curr = $transform(vec4(a_position,0.,1.)).xy;
    vec2 curr = a_position*scale;
    if( a_texcoord.x < 0.0 ) {
        vec2 next = curr + t2*(v_segment.y-v_segment.x);

        rotate( t1, +a_angles.x/2.0, t);
        v_miter.x = signed_distance(curr, curr+t, position);

        rotate( t2, +a_angles.y/2.0, t);
        v_miter.y = signed_distance(next, next+t, position);
    } else {
        vec2 prev = curr - t1*(v_segment.y-v_segment.x);

        rotate( t1, -a_angles.x/2.0,t);
        v_miter.x = signed_distance(prev, prev+t, position);

        rotate( t2, -a_angles.y/2.0,t);
        v_miter.y = signed_distance(curr, curr+t, position);
    }

    if (!closed && v_segment.x <= 0.0) {
        v_miter.x = 1e10;
    }
    if (!closed && v_segment.y >= v_length)
    {
        v_miter.y = 1e10;
    }

    v_texcoord = vec2( u, v*w );

    // Rotation
    float c = cos(theta);
    float s = sin(theta);
    position.xy = vec2( c*position.x - s*position.y,
                        s*position.x + c*position.y );
    // Translation
    position += translate;

    gl_Position = (projection_mat*(modelview_mat))*vec4(position,0.0,1.0);
}

---FRAGMENT SHADER-----------------------------------------------------
#version 120
#ifdef GL_ES
    precision highp float;
#endif

// -----------------------------------------------------------------------------
// Copyright (C) 2013 Nicolas P. Rougier. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY NICOLAS P. ROUGIER ''AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL NICOLAS P. ROUGIER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are
// those of the authors and should not be interpreted as representing official
// policies, either expressed or implied, of Nicolas P. Rougier.
// -----------------------------------------------------------------------------
const float PI = 3.14159265358979323846264;
const float THETA = 15.0 * 3.14159265358979323846264/180.0;

// vec4 stroke(float distance, float linewidth, float antialias, vec4 fg_color)
// {
//     vec4 frag_color;
//     float t = linewidth/2.0 - antialias;
//     float signed_distance = distance;
//     float border_distance = abs(signed_distance) - t;
//     float alpha = border_distance/antialias;
//     alpha = exp(-alpha*alpha);

//     if( border_distance < 0.0 )
//         frag_color = fg_color;
//     else
//         frag_color = vec4(fg_color.rgb, fg_color.a * alpha);

//     return frag_color;
// }

// vec4 stroke(float distance, float linewidth, float antialias, vec4 fg_color, vec4 bg_color)
// {
//     return stroke(distance, linewidth, antialias, fg_color);
// }

float
cap( int type, float dx, float dy, float t )
{
    float d = 0.0;
    dx = abs(dx);
    dy = abs(dy);

    // None
    if      (type == 0)  discard;
    // Round
    else if (type == 1)  d = sqrt(dx*dx+dy*dy);
    // Triangle in
    else if (type == 3)  d = (dx+abs(dy));
    // Triangle out
    else if (type == 2)  d = max(abs(dy),(t+dx-abs(dy)));
    // Square
    else if (type == 4)  d = max(dx,dy);
    // Butt
    else if (type == 5)  d = max(dx+t,dy);

    return d;
}

float
join( in int type, in float d, in vec2 segment, in vec2 texcoord,
      in vec2 miter, in float miter_limit, in float linewidth )
{
    float dx = texcoord.x;

    // Round join
    // --------------------------------
    if( type == 1 )
    {
        if (dx < segment.x) {
            d = max(d,length( texcoord - vec2(segment.x,0.0)));
            //d = length( texcoord - vec2(segment.x,0.0));
        } else if (dx > segment.y) {
            d = max(d,length( texcoord - vec2(segment.y,0.0)));
            //d = length( texcoord - vec2(segment.y,0.0));
        }
    }

    // Bevel join
    // --------------------------------
    else if ( type == 2 )
    {
        if( (dx < segment.x) ||  (dx > segment.y) )
            d = max(d, min(abs(miter.x),abs(miter.y)));
    }

    // Miter limit
    // --------------------------------
    if( (dx < segment.x) ||  (dx > segment.y) )
    {
        d = max(d, min(abs(miter.x),
                       abs(miter.y)) - miter_limit*linewidth/2.0 );
    }

    return d;
}


// // Compute distance to cap
// // ----------------------------------------------------------------------------
// float
// cap( int type, float dx, float dy, float t )
// {
//     float d = 0.0;
//     dx = abs(dx);
//     dy = abs(dy);

//     // None
//     if      (type == 0)  discard;
//     // Round
//     else if (type == 1)  d = sqrt(dx*dx+dy*dy);
//     // Triangle in
//     else if (type == 3)  d = (dx+abs(dy));
//     // Triangle out
//     else if (type == 2)  d = max(abs(dy),(t+dx-abs(dy)));
//     // Square
//     else if (type == 4)  d = max(dx,dy);
//     // Butt
//     else if (type == 5)  d = max(dx+t,dy);

//     return d;
// }


// // Compute distance to join
// // ----------------------------------------------------------------------------
// float
// join( in int type, in float d, in vec2 segment, in vec2 texcoord, in vec2 miter,
//       in float miter_limit, in float linewidth )
// {
//     float dx = texcoord.x;

//     // Round join
//     // --------------------------------
//     if( type == 1 )
//     {
//         if (dx < segment.x) {
//             d = max(d,length( texcoord - vec2(segment.x,0.0)));
//             //d = length( texcoord - vec2(segment.x,0.0));
//         } else if (dx > segment.y) {
//             d = max(d,length( texcoord - vec2(segment.y,0.0)));
//             //d = length( texcoord - vec2(segment.y,0.0));
//         }
//     }

//     // Bevel join
//     // --------------------------------
//     else if ( type == 2 )
//     {
//         if( (dx < segment.x) ||  (dx > segment.y) )
//             d = max(d, min(abs(miter.x),abs(miter.y)));
//     }

//     // Miter limit
//     // --------------------------------
//     if( (dx < segment.x) ||  (dx > segment.y) )
//     {
//         d = max(d, min(abs(miter.x),abs(miter.y)) - miter_limit*linewidth/2.0 );
//     }

//     return d;
// }



// Uniforms
// ------------------------------------
uniform sampler2D u_dash_atlas;

// Varying
// ------------------------------------
varying vec4  v_color;
varying vec2  v_segment;
varying vec2  v_angles;
varying vec2  v_linecaps;
varying vec2  v_texcoord;
varying vec2  v_miter;
varying float v_miter_limit;
varying float v_length;
varying float v_linejoin;
varying float v_linewidth;
varying float v_antialias;
varying float v_dash_phase;
varying float v_dash_period;
varying float v_dash_index;
varying vec2  v_dash_caps;
varying float v_closed;
varying float v_scale;

void main()
{
    // // gl_FragColor = vec4(0.,1.,1.,1.);//my
    // // return;//my

    // // If color is fully transparent we just discard the fragment
    // if( v_color.a <= 0.0 )
    // {
    // 	// gl_FragColor = vec4(1.0,0.0,0.0,.25); return;
    //     discard;
    // }

    // // Test if dash pattern is the solid one (0)
    // bool solid =  (v_dash_index == 0.0);

    // // Test if path is closed
    // bool closed = (v_closed > 0.0);

    // vec4 color = v_color;
    // float dx = v_texcoord.x;
    // float dy = v_texcoord.y;
    // float t = v_linewidth/2.0-v_antialias/v_scale;
    // float width = v_linewidth;
    // float d = 0.0;

    // vec2 linecaps = v_linecaps;
    // vec2 dash_caps = v_dash_caps;
    // float line_start = 0.0;
    // float line_stop  = v_length;


    // // ------------------------------------------------------------------------
    // // Solid line
    // // ------------------------------------------------------------------------
    // if( solid ) {
    //     d = abs(dy);
    //     if( (!closed) && (dx < line_start) )
    //     {
    //         d = cap( int(v_linecaps.x), abs(dx), abs(dy), t );
    //     }
    //     else if( (!closed) &&  (dx > line_stop) )
    //     {
    //         d = cap( int(v_linecaps.y), abs(dx)-line_stop, abs(dy), t );
    //     }
    //     else
    //     {
    //         d = join( int(v_linejoin), abs(dy), v_segment, v_texcoord,
    //                   v_miter, v_miter_limit, v_linewidth );
    //     }

    // // ------------------------------------------------------------------------
    // // Dash line
    // // ------------------------------------------------------------------------
    // } else {
    //     float segment_start = v_segment.x;
    //     float segment_stop  = v_segment.y;
    //     float segment_center= (segment_start+segment_stop)/2.0;
    //     float freq          = v_dash_period*width;
    //     float u = mod( dx + v_dash_phase*width,freq );
    //     vec4 tex = texture2D(u_dash_atlas, vec2(u/freq, v_dash_index));
    //     float dash_center= tex.x * width;
    //     float dash_type  = tex.y;
    //     float _start = tex.z * width;
    //     float _stop  = tex.a * width;
    //     float dash_start = dx - u + _start;
    //     float dash_stop  = dx - u + _stop;

    //     // Compute extents of the first dash (the one relative to v_segment.x)
    //     // Note: this could be computed in the vertex shader
    //     if( (dash_stop < segment_start) && (dash_caps.x != 5.0) )
    //     {
    //         float u = mod(segment_start + v_dash_phase*width, freq);
    //         vec4 tex = texture2D(u_dash_atlas, vec2(u/freq, v_dash_index));
    //         dash_center= tex.x * width;
    //         //dash_type  = tex.y;
    //         float _start = tex.z * width;
    //         float _stop  = tex.a * width;
    //         dash_start = segment_start - u + _start;
    //         dash_stop = segment_start - u + _stop;
    //     }

    //     // Compute extents of the last dash (the one relatives to v_segment.y)
    //     // Note: This could be computed in the vertex shader
    //     else if( (dash_start > segment_stop)  && (dash_caps.y != 5.0) )
    //     {
    //         float u = mod(segment_stop + v_dash_phase*width, freq);
    //         vec4 tex = texture2D(u_dash_atlas, vec2(u/freq, v_dash_index));
    //         dash_center= tex.x * width;
    //         //dash_type  = tex.y;
    //         float _start = tex.z * width;
    //         float _stop  = tex.a * width;
    //         dash_start = segment_stop - u + _start;
    //         dash_stop  = segment_stop - u + _stop;
    //     }

    //     // This test if the we are dealing with a discontinuous angle
    //     bool discontinuous = ((dx <  segment_center) && abs(v_angles.x) > THETA) ||
    //                          ((dx >= segment_center) && abs(v_angles.y) > THETA);
    //     //if( dx < line_start) discontinuous = false;
    //     //if( dx > line_stop)  discontinuous = false;

    //     float d_join = join( int(v_linejoin), abs(dy),
    //                         v_segment, v_texcoord, v_miter, v_miter_limit, v_linewidth );

    //     // When path is closed, we do not have room for linecaps, so we make room
    //     // by shortening the total length
    //     if (closed)
    //     {
    //          line_start += v_linewidth/2.0;
    //          line_stop  -= v_linewidth/2.0;
    //     }

    //     // We also need to take antialias area into account
    //     //line_start += v_antialias;
    //     //line_stop  -= v_antialias;

    //     // Check is dash stop is before line start
    //     if( dash_stop <= line_start )
    //     {
    //         discard;
    //     }
    //     // Check is dash start is beyond line stop
    //     if( dash_start >= line_stop )
    //     {
    //         //discard; //*//
    //     }

    //     // Check if current dash start is beyond segment stop
    //     if( discontinuous )
    //     {
    //         // Dash start is beyond segment, we discard
    //         if( (dash_start > segment_stop) )
    //         {
    //             discard;
    //             //gl_FragColor = vec4(1.0,0.0,0.0,.25); return;
    //         }

    //         // Dash stop is before segment, we discard
    //         if( (dash_stop < segment_start) )
    //         {
    //             discard;
    //             //gl_FragColor = vec4(0.0,1.0,0.0,.25); return;
    //         }

    //         // Special case for round caps (nicer with this)
    //         if( dash_caps.x == 1.0 )
    //         {
    //             if( (u > _stop) && (dash_stop > segment_stop )  && (abs(v_angles.y) < PI/2.0))
    //             {
    //                 discard;
    //             }
    //         }

    //         // Special case for round caps  (nicer with this)
    //         if( dash_caps.y == 1.0 )
    //         {
    //             if( (u < _start) && (dash_start < segment_start )  && (abs(v_angles.x) < PI/2.0))
    //             {
    //                 discard;
    //             }
    //         }

    //         // Special case for triangle caps (in & out) and square
    //         // We make sure the cap stop at crossing frontier
    //         if( (dash_caps.x != 1.0) && (dash_caps.x != 5.0) )
    //         {
    //             if( (dash_start < segment_start )  && (abs(v_angles.x) < PI/2.0) )
    //             {
    //                 float a = v_angles.x/2.0;
    //                 float x = (segment_start-dx)*cos(a) - dy*sin(a);
    //                 float y = (segment_start-dx)*sin(a) + dy*cos(a);
    //                 if( x > 0.0 ) discard;
    //                 // We transform the cap into square to avoid holes
    //                 dash_caps.x = 4.0;
    //             }
    //         }

    //         // Special case for triangle caps (in & out) and square
    //         // We make sure the cap stop at crossing frontier
    //         if( (dash_caps.y != 1.0) && (dash_caps.y != 5.0) )
    //         {
    //             if( (dash_stop > segment_stop )  && (abs(v_angles.y) < PI/2.0) )
    //             {
    //                 float a = v_angles.y/2.0;
    //                 float x = (dx-segment_stop)*cos(a) - dy*sin(a);
    //                 float y = (dx-segment_stop)*sin(a) + dy*cos(a);
    //                 if( x > 0.0 ) discard;
    //                 // We transform the caps into square to avoid holes
    //                 dash_caps.y = 4.0;
    //             }
    //         }
    //     }

    //     // Line cap at start
    //     if( (dx < line_start) && (dash_start < line_start) && (dash_stop > line_start) )
    //     {
    //         d = cap( int(linecaps.x), dx-line_start, dy, t);
    //     }
    //     // Line cap at stop
    //     else if( (dx > line_stop) && (dash_stop > line_stop) && (dash_start < line_stop)  )
    //     {
    //         d = cap( int(linecaps.y), dx-line_stop, dy, t);
    //     }
    //     // Dash cap left
    //     else if( dash_type < 0.0 )
    //     {
    //         d = cap( int(dash_caps.y), abs(u-dash_center), dy, t);
    //         if( (dx > line_start) && (dx < line_stop) )
    //             d = max(d,d_join);
    //     }
    //     // Dash cap right
    //     else if( dash_type > 0.0 )
    //     {
    //         d = cap( int(dash_caps.x), abs(dash_center-u), dy, t);
    //         if( (dx > line_start) && (dx < line_stop) )
    //             d = max(d,d_join);
    //     }
    //     // Dash body (plain)
    //     else if( dash_type == 0.0 )
    //     {
    //         d = abs(dy);
    //     }

    //     // Line join
    //     if( (dx > line_start) && (dx < line_stop))
    //     {
    //         if( (dx <= segment_start) && (dash_start <= segment_start)
    //             && (dash_stop >= segment_start) )
    //         {
    //             d = d_join;
    //             // Antialias at outer border
    //             float angle = PI/2.+v_angles.x;
    //             float f = abs( (segment_start - dx)*cos(angle) - dy*sin(angle));
    //             d = max(f,d);
    //         }
    //         else if( (dx > segment_stop) && (dash_start <= segment_stop)
    //                  && (dash_stop >= segment_stop) )
    //         {
    //             d = d_join;
    //             // Antialias at outer border
    //             float angle = PI/2.+v_angles.y;
    //             float f = abs((dx - segment_stop)*cos(angle) - dy*sin(angle));
    //             d = max(f,d);
    //         }
    //         else if( dx < (segment_start - v_linewidth/2.))
    //         {
    //             discard;
    //         }
    //         else if( dx > (segment_stop + v_linewidth/2.))
    //         {
    //             discard;
    //         }
    //     }
    //     else if( dx < (segment_start - v_linewidth/2.))
    //     {
    //         discard;
    //     }
    //     else if( dx > (segment_stop + v_linewidth/2.))
    //     {
    //         discard;
    //     }
    // }


    // // Distance to border
    // // ------------------------------------------------------------------------
    // d = d - t;
    // if( d < 0.0 )
    // {
    //     gl_FragColor = color; // vec4(.3,.7,.0,1.); //
    // }
    // else
    // {
    //     d /= v_antialias;
    //     // alpha = clamp(1. - (d - v_width / 2.) * v_scale, 0., 1.);
    //     // gl_FragColor = vec4(color.xyz, color.a * clamp(1. - d * v_scale, 0., 1.));//clamp(1. - d * v_scale * 0., 0., 1.)); //exp(-d*d)*color.a);
    //     gl_FragColor = vec4(color.xyz, color.a * clamp(1. - d * v_scale, 0., 1.));
    // }
    // // d = d - t;
    // // if( d < 0.0 )
    // // {
    // //     gl_FragColor = color; // vec4(.3,.7,.0,1.); //
    // // }
    // // else
    // // {
    // //     d /= v_antialias;
    // //     gl_FragColor = vec4(color.xyz, exp(-d*d)*color.a);
    // // }
    // gl_FragColor = v_color; return;
    // vec4 color = v_color;

    // If color is fully transparent we just discard the fragment
    if( v_color.a <= 0.0 ) {
        discard;
    }

    // Test if dash pattern is the solid one (0)
    bool solid = (v_dash_index == 0.0);

    float dx = v_texcoord.x;
    float dy = v_texcoord.y;
    // float t = v_linewidth/2.0-v_antialias;
    float t = v_linewidth/2.0-v_antialias/v_scale;
    float width = v_linewidth;
    float d = 0.0;

    vec2 linecaps = v_linecaps;
    vec2 dash_caps = v_dash_caps;
    float line_start = 0.0;
    float line_stop  = v_length;
    // Test if path is closed
    bool closed = (v_closed > 0.0);

    // ------------------------------------------------------------------------
    // Solid line
    // ------------------------------------------------------------------------
    if( solid ) {
        d = abs(dy);

        if( (!closed) && (dx < line_start) )
        {
            d = cap( int(v_linecaps.x), abs(dx), abs(dy), t );
        }
        else if( (!closed) &&  (dx > line_stop) )
        {
            d = cap( int(v_linecaps.y), abs(dx)-line_stop, abs(dy), t );
        }
        else
        {
            d = join( int(v_linejoin), abs(dy), v_segment, v_texcoord,
                      v_miter, v_miter_limit, v_linewidth );
        }

    // ------------------------------------------------------------------------
    // Dash line
    // ------------------------------------------------------------------------
    } else {
        float segment_start = v_segment.x;
        float segment_stop  = v_segment.y;
        float segment_center = (segment_start+segment_stop)/2.0;
        float freq = v_dash_period*width;
        float u = mod( dx + v_dash_phase*width,freq );
        vec4 tex = texture2D(u_dash_atlas, vec2(u/freq, v_dash_index));
        float dash_center= tex.x * width;
        float dash_type  = tex.y;
        float _start = tex.z * width;
        float _stop  = tex.a * width;
        float dash_start = dx - u + _start;
        float dash_stop  = dx - u + _stop;

        // This test if the we are dealing with a discontinuous angle
        bool discont = ((dx <  segment_center) && abs(v_angles.x) > THETA) ||
                       ((dx >= segment_center) && abs(v_angles.y) > THETA);
        if( dx < line_start) discont = false;
        if( dx > line_stop)  discont = false;

        // When path is closed, we do not have room for linecaps, so we make
        // room by shortening the total length
        if (closed){
            line_start += v_linewidth/2.0;
            line_stop  -= v_linewidth/2.0;
            linecaps = v_dash_caps;
        }


        // Check is dash stop is before line start
        if( dash_stop <= line_start )
        {
            discard;
        }
        // Check is dash start is beyond line stop
        if( dash_start >= line_stop )
        {
            discard;
        }

        // Check if current pattern start is beyond segment stop
        if( discont )
        {
            // Dash start is beyond segment, we discard
            if( dash_start > segment_stop )
            {
                discard;
            }

            // Dash stop is before segment, we discard
            if( dash_stop < segment_start )
            {
                discard;
            }

            // Special case for round caps (nicer with this)
            if( (u > _stop) && (dash_stop > segment_stop ) &&
                (abs(v_angles.y) < PI/2.0))
            {
                if( dash_caps.x == 1.0) discard;
            }
            // Special case for round caps  (nicer with this)
            else if( (u < _start) && (dash_start < segment_start ) &&
                     (abs(v_angles.x) < PI/2.0))
            {
                if( dash_caps.y == 1.0) discard;
            }

            // Special case for triangle caps (in & out) and square
            // We make sure the cap stop at crossing frontier
            if( (dash_caps.x != 1.0) && (dash_caps.x != 5.0) )
            {
                if( (dash_start < segment_start ) &&
                    (abs(v_angles.x) < PI/2.0) )
                {
                    float a = v_angles.x/2.0;
                    float x = (segment_start-dx)*cos(a) - dy*sin(a);
                    float y = (segment_start-dx)*sin(a) + dy*cos(a);
                    if( (x > 0.0) ) discard;
                    // We transform the cap into square to avoid holes
                    dash_caps.x = 4.0;
                }
            }
            // Special case for triangle caps (in & out) and square
            // We make sure the cap stop at crossing frontier
            if( (dash_caps.y != 1.0) && (dash_caps.y != 5.0) )
            {
                if( (dash_stop > segment_stop ) &&
                    (abs(v_angles.y) < PI/2.0) )
                {
                    float a = v_angles.y/2.0;
                    float x = (dx-segment_stop)*cos(a) - dy*sin(a);
                    float y = (dx-segment_stop)*sin(a) + dy*cos(a);
                    if( (x > 0.0) ) discard;
                    // We transform the caps into square to avoid holes
                    dash_caps.y = 4.0;
                }
            }
        }

        // Line cap at start
        if( (dx < line_start) && (dash_start < line_start) &&
            (dash_stop > line_start) )
        {
            d = cap( int(linecaps.x), dx-line_start, dy, t);
        }
        // Line cap at stop
        else if( (dx > line_stop) && (dash_stop > line_stop) &&
                 (dash_start < line_stop)  )
        {
            d = cap( int(linecaps.y), dx-line_stop, dy, t);
        }
        // Dash cap left
        else if( dash_type < 0.0 )
        {
            float u = max( u-dash_center , 0.0 );
            d = cap( int(dash_caps.y), abs(u), dy, t);
        }
        // Dash cap right
        else if( dash_type > 0.0 )
        {
            float u = max( dash_center-u, 0.0 );
            d = cap( int(dash_caps.x), abs(u), dy, t);
        }
        // Dash body (plain)
        else if( dash_type == 0.0 )
        {
            d = abs(dy);
        }

        // Antialiasing at segment angles region
        if( discont )
        {
            if( dx < segment_start )
            {
                // For sharp angles, we do not enforce cap shape
                if( (dash_start < segment_start ) &&
                    (abs(v_angles.x) > PI/2.0))
                {
                    d = abs(dy);
                }
                // Antialias at outer border
                dx = segment_start - dx;
                float angle = PI/2.+v_angles.x;
                float f = abs( dx*cos(angle) - dy*sin(angle));
                d = max(f,d);
            }
            else if( (dx > segment_stop) )
            {
                // For sharp angles, we do not enforce cap shape
                if( (dash_stop > segment_stop ) &&
                    (abs(v_angles.y) > PI/2.0) )
                {
                    d = abs(dy);
                }
                // Antialias at outer border
                dx = dx - segment_stop;
                float angle = PI/2.+v_angles.y;
                float f = abs( dx*cos(angle) - dy*sin(angle));
                d = max(f,d);
            }
        }

        // Line join
        //if( (dx > line_start) && (dx < line_stop) )
        {
            d = join( int(v_linejoin), d, v_segment, v_texcoord,
                      v_miter, v_miter_limit, v_linewidth );
        }
    }


    // Distance to border
    // ------------------------------------------------------------------------
    d = d - t;
    if( d < 0.0 )
    {
        gl_FragColor = v_color;
    }
    else
    {
        d /= v_antialias;
        //gl_FragColor = vec4(v_color.xyz, exp(-d*d)*v_color.a);
        //alpha = clamp(1. - (d - v_width / 2.) * v_scale, 0., 1.);
        gl_FragColor = vec4(v_color.xyz, v_color.a * clamp(1. - d * v_scale, 0., 1.));
    }
}
