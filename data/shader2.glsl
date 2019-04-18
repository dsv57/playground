---VERTEX SHADER-------------------------------------------------------
#version 120
#ifdef GL_ES
    precision highp float;
#endif

const float M_PI = 3.14159265358979323846264338327950288;
const float M_PI_2 = 1.57079632679489661923132169163975144;

/* Outputs to the fragment shader */
varying vec2  v_position;
varying vec2  v_tex_coord0;
varying vec2  v_tex_coord1;
varying vec2  v_size;
varying float v_radius;
varying float v_width;
varying vec4  v_stroke;
varying vec4  v_fill;
varying float v_scale;

/* vertex attributes */
attribute vec2  size;
attribute vec2  center;
attribute float radius;
attribute float width;
attribute vec4  stroke;
attribute vec4  fill;
attribute vec4  transform;
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
  v_tex_coord0 = tex_coords0;
  v_tex_coord1 = tex_coords1;
  v_radius = radius; //(1.+cos(time*3.))*5.;
  v_width = width; //(1.+cos(time*3.))*5.;
  v_stroke = stroke;
  v_fill = fill;
  mat2 transform = mat2(transform);
  vec2 size2 = size;
  v_scale = scale * max(
    sqrt(transform[0][0] * transform[0][0] + transform[1][0] * transform[1][0]),
    sqrt(transform[0][1] * transform[0][1] + transform[1][1] * transform[1][1])
  );
  // float c = cos(M_PI_2 / 2.);
  // float s = sin(M_PI_2 / 2.);
  // transform *= mat2(c, s, -s, c);

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
varying vec2  v_tex_coord0;
varying vec2  v_tex_coord1;
varying vec2  v_size;
varying float v_radius;
varying float v_width;
varying vec4  v_stroke;
varying vec4  v_fill;
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

float SDF_round_box(vec2 p, vec2 size, float radius)
{
    vec2 d = abs(p) - size + vec2(radius);
    return min(max(d.x, d.y),0.0) + length(max(d, 0.0)) - radius;
}

// /* Antialias */
// if (d > b+1.) discard;
// a *= min(1. - (abs(d)-b), 1.);
void main() {
    vec4 clr;
    float alpha;
    float d = SDF_round_box(v_position, v_size / 2., v_radius);

    if (v_width > 0.01) { /* Stroke */
        alpha = clamp(1. - (abs(d) - v_width / 2.) * v_scale, 0., 1.);
        if (d < 0.) {
            clr = mix(v_fill, v_stroke, alpha); //vec4(.5,.0,.0,.5);//
            // clr.a = .7;
        } else {
            clr = v_stroke * srgb_to_linear(texture2D(texture1, v_tex_coord1));
            clr.a *= alpha;
        }
    } else { /* Fill */
        alpha = clamp(1. - (d - v_width / 2.) * v_scale, 0., 1.);
        clr = v_fill * srgb_to_linear(texture2D(texture0, v_tex_coord0));
        clr.a *= alpha;
    }

    // /* Fill */
    // if (stroke && (d < -v_width / 2.)) {
    //     clr = mix(v_fill, v_stroke, alpha);
    // } else {
    //     if (d > v_width / 2.)
    //       clr.a *= alpha;
    // }

    gl_FragColor = clr;
}
