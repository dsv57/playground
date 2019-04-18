---VERTEX SHADER-------------------------------------------------------
#version 120
#ifdef GL_ES
    precision highp float;
#endif

const float M_PI = 3.14159265358979323846264338327950288;
const float M_PI_2 = 1.57079632679489661923132169163975144;

/* Outputs to the fragment shader */
varying vec2  tex_coord0;
varying vec2  tex_coord1;
varying vec2  v_position;
varying vec2  v_size;
varying float v_width;
varying vec4  v_stroke;
varying vec4  v_fill;
varying float v_angle_start;
varying float v_angle_end;
varying float v_scale;

/* vertex attributes */
attribute vec2  center;
attribute float width;
attribute vec4  stroke;
attribute vec4  fill;
attribute float angle_start;
attribute float angle_end;
attribute vec4  transform;
attribute vec2  size;
attribute vec2  tex_coords0;
attribute vec2  tex_coords1;

/* uniform variables */
uniform mat4  modelview_mat;
uniform mat4  projection_mat;
// uniform vec4  color;
uniform float scale;

uniform vec2  resolution;
uniform float time;

void main (void) {
  tex_coord0 = tex_coords0;
  tex_coord1 = tex_coords1;
  v_width = width;//(1.+cos(time*8.))*5.;
  v_stroke = stroke;
  v_fill = fill;
  v_angle_start = angle_start;// + time;
  v_angle_end = angle_end;// + time;
  mat2 transform = mat2(transform);
  vec2 size2 = size;
  v_scale = scale * max(
    sqrt(transform[0][0] * transform[0][0] + transform[1][0] * transform[1][0]),
    sqrt(transform[0][1] * transform[0][1] + transform[1][1] * transform[1][1])
  );

  // if (abs(size.y) > abs(size.x)) {
  //   transform *= mat2(0, -1., 1., 0.);
  //   size2.x = size.y;
  //   size2.y = size.x;
  // }
  v_size = abs(size2);// * (1.+sin(time+center.x*center.y));
  v_position = size2 / 2. + sign(size2) * (1. / scale + v_width / 2.);

  gl_Position = projection_mat * modelview_mat * vec4(center + transform * v_position, 0.0, 1.0);
}

---FRAGMENT SHADER-----------------------------------------------------
#version 120
#ifdef GL_ES
    precision highp float;
#endif

const float M_PI = 3.14159265358979323846264338327950288;
const float M_PI_2 = 1.57079632679489661923132169163975144;

/* Outputs from the vertex shader */
varying vec2  v_position;
varying vec2  tex_coord0;
varying vec2  tex_coord1;
varying vec2  v_size;
varying float v_width;
varying vec4  v_stroke;
varying vec4  v_fill;
varying float v_angle_start;
varying float v_angle_end;
varying float v_scale;

/* uniform texture samplers */

/* custom one */
uniform vec2 resolution;
uniform float time;
uniform float scale;
uniform sampler2D texture0;
uniform sampler2D texture1;

// vec4 LinearTosRGB( in vec4 value ) {
//     return vec4(mix(
//         pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ),
//         value.rgb * 12.92,
//         vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.w );
// }

vec4 srgb_to_linear(vec4 srgb){
    vec3 cutoff = vec3(lessThan(srgb.rgb, vec3(12.92 * 0.0031308)));
    vec3 higher = pow((srgb.rgb + 0.055) / 1.055, vec3(2.4));
    vec3 lower = srgb.rgb / vec3(12.92);
    return vec4(mix(higher, lower, cutoff), srgb.w);
}

float SDF_fake_ellipse(vec2 p, vec2 size) {
  float a = 1.0;
  float b = size.x/size.y;
  float r = 0.5*max(size.x,size.y);
  float f = length(p*vec2(a,b));
  return f*(f-r)/length(p*vec2(a*a,b*b));
}

float SDF_ellipse(vec2 p, vec2 ab) {
  // The function does not like circles
  if (ab.x == ab.y) //ab.x = ab.x*0.9999;
    return length(p) - ab.x;

  p = abs( p ); if( p.x > p.y ){ p=p.yx; ab=ab.yx; }
  float l = ab.y*ab.y - ab.x*ab.x;
  float m = ab.x*p.x/l;
  float n = ab.y*p.y/l;
  float m2 = m*m;
  float n2 = n*n;
  float c = (m2 + n2 - 1.0)/3.0;
  float c3 = c*c*c;
  float q = c3 + m2*n2*2.0;
  float d = c3 + m2*n2;
  float g = m + m*n2;
  float co;

  if( d<0.0 ) {
      float p = acos(q/c3)/3.0;
      float s = cos(p);
      float t = sin(p)*sqrt(3.0);
      float rx = sqrt( -c*(s + t + 2.0) + m2 );
      float ry = sqrt( -c*(s - t + 2.0) + m2 );
      co = ( ry + sign(l)*rx + abs(g)/(rx*ry) - m)/2.0;
  } else {
      float h = 2.0*m*n*sqrt( d );
      float s = sign(q+h)*pow( abs(q+h), 1.0/3.0 );
      float u = sign(q-h)*pow( abs(q-h), 1.0/3.0 );
      float rx = -s - u - c*4.0 + 2.0*m2;
      float ry = (s - u)*sqrt(3.0);
      float rm = sqrt( rx*rx + ry*ry );
      float p = ry/sqrt(rm-rx);
      co = (p + 2.0*g/rm - m)/2.0;
  }
  float si = sqrt( 1.0 - co*co );
  vec2 r = vec2( ab.x*co, ab.y*si );
  return length(r - p ) * sign(p.y-r.y);
}

float SDF_sector(vec2 p, float a, float b) {
    float p1 = dot(p, vec2(sin(a), -cos(a)));
    float p2 = dot(p, vec2(-sin(b), cos(b)));
    return b - a < M_PI ? max(p1, p2) : min(p1, p2);
}

// /* Antialias */
// if (d > b+1.) discard;
// a *= min(1. - (abs(d)-b), 1.);
void main() {
    vec4 clr;
    float alpha;
    float d;
    // if ((v_width*scale > 10.) || (v_size.x / v_size.y > 3.)) {
    d = SDF_ellipse(v_position, v_size / 2.);
    // } else {
    //     d = SDF_fake_ellipse(v_position, v_size);
    // }
    if (v_angle_end - v_angle_start < 2 *M_PI) {
        d = max(d, SDF_sector(v_position, v_angle_start, v_angle_end));
    }

    if (v_width > 0.01) { /* Stroke */
        alpha = clamp(1. - (abs(d) - v_width / 2.) * v_scale, 0., 1.);
        clr = v_stroke * srgb_to_linear(texture2D(texture1, tex_coord1));
        if (d < 0.) {
            clr = mix(v_fill, clr, alpha);
        } else {
            // clr = v_stroke;// * srgb_to_linear(texture2D(texture1, tex_coord1));
            clr.a *= alpha;
        }
    } else { /* Fill */
        alpha = clamp(1. - (d - v_width / 2.) * v_scale, 0., 1.);
        clr = v_fill * srgb_to_linear(texture2D(texture0, tex_coord0));
        clr.a *= alpha;
    }

    gl_FragColor = clr;
}
