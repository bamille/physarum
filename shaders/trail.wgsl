// Pass 3 — the trail map.
//
// One quad spanning the simulation bounds in world space, pushed through the
// same camera the agents use, so the trail and the agents can never drift out
// of alignment. The fragment shader reads the trail buffer directly —
// read-only storage is visible to every stage, so no texture is involved.

struct Camera {
    view_proj: mat4x4<f32>,
    right:     vec4<f32>,
    up:        vec4<f32>,
    params:    vec4<f32>,   // x = agent radius
};

struct SimParams {
    grid:         vec2<u32>,
    n_agents:     u32,
    frame:        u32,

    world_half:   vec2<f32>,
    dt:           f32,
    sensor_angle: f32,

    sensor_dist:  f32,
    turn_speed:   f32,
    move_speed:   f32,
    sensitivity:  f32,

    deposit:      f32,
    decay_rate:   f32,
    diffuse_rate: f32,
    row_width:    u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> trail: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Same triangle-strip corner order as the billboards:
    //   0 -> (-1,-1)   1 -> (-1,+1)   2 -> (+1,-1)   3 -> (+1,+1)
    let corner = vec2<f32>(
        f32(vi >> 1u) * 2.0 - 1.0,
        f32(vi & 1u) * 2.0 - 1.0,
    );

    let world = corner * params.world_half;

    var out: VsOut;
    out.position = camera.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.uv = corner * 0.5 + vec2<f32>(0.5);   // 0..1 across the world
    return out;
}

// Per-channel tint. Channel 0 is the only one in use until there are multiple
// species; the others are here so adding one is a colour choice, not a
// rewrite.
const TINT0 = vec3<f32>(0.20, 0.85, 1.00);   // cyan
const TINT1 = vec3<f32>(1.00, 0.42, 0.25);   // orange
const TINT2 = vec3<f32>(0.55, 1.00, 0.35);   // green
const TINT3 = vec3<f32>(0.95, 0.80, 0.30);   // amber

/// Brightness per unit of trail. Trail values are unbounded — a cell that
/// stays busy settles around deposit/(1 - decay per step) — so the tonemap
/// below is what keeps a saturated cell from clipping to flat white.
const EXPOSURE: f32 = 0.35;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let g = vec2<f32>(params.grid);
    let cell = vec2<u32>(clamp(in.uv * g, vec2<f32>(0.0), g - vec2<f32>(1.0)));
    let t = trail[cell.y * params.grid.x + cell.x];

    let linear = TINT0 * t.x + TINT1 * t.y + TINT2 * t.z + TINT3 * t.w;

    // Exponential tonemap: linear for small values, asymptotic to 1 for large
    // ones. Nothing ever clips, and faint trails stay visible.
    let mapped = vec3<f32>(1.0) - exp(-linear * EXPOSURE);

    return vec4<f32>(mapped, 1.0);
}
