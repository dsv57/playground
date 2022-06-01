$HEADER$

vec3 to_linear(vec3 srgb){
    // vec3 cutoff = vec3(lessThan(srgb, vec3(12.92 * 0.0031308)));
    // vec3 higher =  pow((srgb + 0.055) / 1.055, vec3(2.4));
    // vec3 lower = srgb / vec3(12.92);
    // return mix(higher, lower, cutoff);
    return mix( pow( srgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), srgb * 0.0773993808, vec3( lessThanEqual( srgb, vec3( 0.04045 ) ) ) );
}

void main() {
    vec4 srgb = frag_color * texture2D(texture0, tex_coord0);
    gl_FragColor = vec4(to_linear(srgb.rgb), srgb.a);
}
