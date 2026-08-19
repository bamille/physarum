// Pass 3 — agent billboards.
//
// One camera-facing quad per agent. No vertex buffer: the draw is
// draw(0..4, 0..n_agents) over a triangle strip, and both the corner and the
// agent are looked up from builtins.

struct Camera {
    view_proj: mat4x4<f32>,  // proj * view, world -> clip
    right:     vec4<f32>,    // world-space camera basis, w = 0
    up:        vec4<f32>,
    params:    vec4<f32>,    // x = agent radius, y = speed_scale, z = dt
};

struct Agent {
    pos:     vec2<f32>,      // world XY; agents live on the z = 0 plane
    heading: vec2<f32>,      // unit vector
    speed:   f32,
    species: u32,            // which trail channel
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> agents: array<Agent>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    // Quad-local coordinate, -1..1 on both axes. Interpolated across the
    // quad, so the fragment shader can tell how far it is from the centre.
    @location(0) corner: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VsOut {
    // Triangle-strip corners, in the order the strip wants them:
    //   0 -> (-1,-1)   1 -> (-1,+1)   2 -> (+1,-1)   3 -> (+1,+1)
    let corner = vec2<f32>(
        f32(vi >> 1u) * 2.0 - 1.0,
        f32(vi & 1u) * 2.0 - 1.0,
    );

    let agent = agents[ii];
    let radius = camera.params.x;

    // Offset along the *camera's* right/up rather than world X/Y. That is what
    // keeps the quad facing the viewer when the camera moves off-axis.
    let centre = vec3<f32>(agent.pos, 0.0);
    let world = centre
        + camera.right.xyz * corner.x * radius
        + camera.up.xyz * corner.y * radius;

    var out: VsOut;
    out.position = camera.view_proj * vec4<f32>(world, 1.0);
    out.corner = corner;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // The quad is square; the dot is the inscribed circle.
    let d = length(in.corner);
    if (d > 1.0) {
        discard;
    }

    // Blending is additive with SrcAlpha as the source factor, so alpha is
    // brightness here, not transparency. Falling off to 0 at the rim gives a
    // soft dot instead of a hard-edged disc.
    let falloff = 1.0 - d;
    return vec4<f32>(0.45, 0.85, 1.0, falloff * falloff);
}
