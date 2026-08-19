// Pass 2 — diffuse + decay.
//
// One invocation per trail cell. Blurs toward the neighbourhood mean and fades
// the whole map. This is the pass that turns individual deposits into the
// continuous field the agents can actually follow: without it, a sensor 9
// cells ahead almost never lands exactly on a cell some other agent touched,
// and no structure forms.

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

// Stencil: every cell reads its neighbours, so this cannot be done in place —
// hence the ping-pong. `src` is read-only, which is also what lets the driver
// know there is no aliasing here.
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

// Half-width of the blur kernel, in cells. 1 gives the textbook 3x3; 2 gives
// 5x5 and spreads noticeably further per step, at 25 samples per cell instead
// of 9. This is the knob to reach for when trails look too thin and stringy —
// raising `diffuse_rate` past 1.0 does nothing, but widening the kernel keeps
// working.
const BLUR_RADIUS: i32 = 1;

@compute @workgroup_size(8, 8, 1)
fn diffuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Guard both axes — the dispatch is rounded up on each.
    if (gid.x >= params.grid.x || gid.y >= params.grid.y) {
        return;
    }

    let w = i32(params.grid.x);
    let h = i32(params.grid.y);
    let x = i32(gid.x);
    let y = i32(gid.y);

    // Box blur. Wrapping at the edges matches the agents' torus, so trails run
    // off one side and back in the other instead of piling up against a rim.
    var sum = vec4<f32>(0.0);
    var count = 0.0;
    for (var dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; dy++) {
        for (var dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; dx++) {
            let sx = (x + dx + w) % w;
            let sy = (y + dy + h) % h;
            sum += src[u32(sy * w + sx)];
            count += 1.0;
        }
    }
    let mean = sum / count;

    let idx = u32(y * w + x);
    let blurred = mix(src[idx], mean, clamp(params.diffuse_rate, 0.0, 1.0));

    // Multiplicative decay, expressed as a half-life in seconds rather than a
    // per-frame factor, so the look does not change with frame rate.
    dst[idx] = blurred * exp(-params.decay_rate * params.dt);
}
