$HEADER$

vec3 fromLinear(vec3 linearRGB) {
    vec3 cutoff = vec3(lessThan(linearRGB, vec3(0.0031308)));
    vec3 higher = vec3(1.055)*pow(linearRGB, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = linearRGB * vec3(12.92);
    return mix(higher, lower, cutoff);
}

void main() {
    vec4 linrgb = frag_color * texture2D(texture0, tex_coord0)*0.5;
    gl_FragColor = vec4(fromLinear(linrgb.rgb), linrgb.a);
}
