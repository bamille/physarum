# Slime mold: the algorithm and where to put it on the GPU

Physarum-style agent simulation, following Jeff Jones' 2010 model. This is a
design note, not code — it exists so that when you sit down offline you are
deciding parameters, not architecture.

---

## 1. The model, in full

State is two things:

- **Agents.** Each has a position (2D, float) and a heading angle. That is all.
  10k is enough to look alive; 1M is where it gets good and is well within
  reach.
- **Trail map.** A 2D scalar field, same aspect as the window. Call it
  `width × height` floats.

Each step, in order:

1. **Sense.** Each agent samples the trail map at three points: straight ahead,
   and ahead-left / ahead-right rotated by `sensor_angle`, all at distance
   `sensor_distance`.
2. **Steer.** If the centre sample is strongest, go straight. If left is
   strongest, rotate left by `turn_speed`. If right, rotate right. If both
   sides beat the centre, rotate randomly left or right. (That last case is
   what breaks symmetry and produces the networks — do not skip it.)
3. **Move.** `pos += vec2(cos(angle), sin(angle)) * move_speed`. On hitting a
   boundary, either wrap or reflect with a randomised angle. Wrapping gives a
   torus-like network; reflecting gives visible walls. Wrap first.
4. **Deposit.** Add `deposit_amount` to the trail map cell under the agent.
5. **Diffuse.** Replace each trail cell with the mean of its 3×3 neighbourhood.
6. **Decay.** Multiply every trail cell by `decay_rate` (or subtract a constant;
   multiplicative is better behaved).

Steps 5 and 6 are one pass. Steps 1–4 are one pass.

The entire character of the result lives in the ratio between `sensor_distance`,
`move_speed`, and `decay_rate`. This is why the egui sliders are in the
dependency list.

### Starting parameters that are known to work

| Parameter | Value | Notes |
|---|---|---|
| `sensor_angle` | 22.5° | in radians in the shader |
| `sensor_distance` | 9.0 px | the big one; try 3 and 25 |
| `turn_speed` | 40°/s | scale by `dt` |
| `move_speed` | 60 px/s | scale by `dt` |
| `deposit_amount` | 5.0 | |
| `decay_rate` | 0.9 /frame | or `exp(-k*dt)` for frame-rate independence |
| `diffuse_rate` | 1.0 | lerp weight toward the 3×3 mean |
| agent count | 100k–1M | |

Initial agent placement matters more than it sounds: uniform random over the
whole screen gives a fairly boring even mesh. A filled circle with headings
pointing inward, or a ring with headings tangential, gives much more structure.

---

## 2. Resource layout

Two designs work. Pick the first.

### Design A — trail as ping-ponged storage buffers (recommended)

```
agents      : storage buffer, array<Agent>, read_write     [STORAGE | COPY_DST]
trail[0..2] : storage buffer, array<f32>, ping-ponged      [STORAGE | COPY_DST]
params      : uniform buffer, Params                       [UNIFORM | COPY_DST]
```

| Pass | Dispatch over | Reads | Writes |
|---|---|---|---|
| 1. agents | `agent_count` | `trail[cur]` (read_write, same binding), `params` | `agents`, `trail[cur]` |
| 2. diffuse + decay | `width × height` | `trail[cur]` (read), `params` | `trail[next]` |
| 3. render | fullscreen triangle | `trail[next]` (read, **fragment stage**) | swapchain |

Why this is the easy path:

- No texture format constraints at all. No `STORAGE_BINDING` format table, no
  sRGB traps, no `bytes_per_row` padding when you want to dump a frame.
- The diffuse pass is a textbook stencil, so ping-pong applies cleanly, and it
  is the only pass that reads neighbours.
- Pass 3 reading a storage buffer from the fragment shader is legal
  (read-only storage buffers are visible to every stage) and means you never
  create a texture at all. Cost: no hardware bilinear filtering — at one trail
  cell per pixel you do not want any.

The subtle bit is pass 1 reading and writing `trail[cur]` in the same dispatch.
That *is* a cross-workgroup race — agent A's sense may or may not see agent B's
deposit from the same step. For this simulation it is invisible: the deposit is
small, the diffuse pass smears it anyway, and a one-step-stale read is
physically as defensible as a fresh one. Do not "fix" it by adding a third
buffer until you have looked at the output.

### Design B — trail as ping-ponged storage textures

```
trail[0..2] : texture_storage_2d<rgba16float, ...>   [STORAGE_BINDING | TEXTURE_BINDING]
```

Worth it if you want multi-channel trails (species with different colours — the
classic three-species variant), or hardware bilinear sampling of the trail for
the sensors. Costs you the format table, and pass 1 can no longer read and
write the same texture, because binding one subresource as writable storage
plus anything else in a single pass is a validation error. You need a third
texture or a split pass.

Start with A. Move to B when you want colour species.

---

## 3. Deposits and the missing `atomic<f32>`

Many agents land on the same cell in one step. In Design A pass 1, that is
`trail[idx] += deposit` from multiple invocations with no ordering — a
read-modify-write race, so some deposits are simply lost.

Three responses, in order of how much they cost you:

1. **Accept it.** With 100k+ agents over a 1M-cell grid, collisions are rare and
   the diffuse pass launders the error. Visually identical. This is what most
   implementations do, usually without realising it.
2. **Fixed-point atomics.** Store the trail as `array<atomic<u32>>`, deposit
   with `atomicAdd(&trail[idx], u32(deposit * 1024.0))`, and divide by 1024 on
   read. Exact, cheap, and caps your dynamic range — pick the scale so
   `max_trail * 1024 < 2^32`.
3. **A separate scatter pass with a CAS loop on float bits.** Correct, slow,
   and unnecessary here.

There is no `atomic<f32>` in WGSL. If you find a tutorial using one, it is
either CUDA or wrong.

---

## 4. Dispatch shapes

- Agent pass: 1D, `@workgroup_size(64)`, `dispatch(ceil(n/64))`. Guard with
  `if (id.x >= params.agent_count) { return; }` — the dispatch is rounded up.
- Diffuse pass: 2D, `@workgroup_size(8, 8)`, `dispatch(ceil(w/8), ceil(h/8))`.
  Same bounds guard on both axes.
- **65,535 workgroups per dimension.** At `@workgroup_size(64)` a 1D dispatch
  caps at 4.19M agents. Past that you need a 2D dispatch and manual index
  reconstruction (`id.y * width + id.x`).

---

## 5. Suggested file layout

Nothing here exists yet; this is the shape to grow into.

```
src/
  main.rs        window + event loop (winit ApplicationHandler)
  gpu.rs         instance/adapter/device/queue/surface setup
  sim.rs         buffers, bind groups, pipelines, the step() function
  params.rs      Params struct (#[repr(C)], Pod) + egui slider panel
  ui.rs          egui-wgpu integration
shaders/
  agents.wgsl    pass 1
  diffuse.wgsl   pass 2
  render.wgsl    pass 3 — fullscreen triangle + colormap
```

The fullscreen triangle: no vertex buffer, `draw(0..3)`, and derive the clip
position from `@builtin(vertex_index)`. Three vertices covering the screen, not
two triangles — it avoids the diagonal seam where derivatives go wrong.

---

## 6. Order to build it in

1. Window opens, clears to a colour. (winit + wgpu surface)
2. Fullscreen triangle renders a gradient from a hardcoded shader.
3. Trail buffer exists, initialised to a test pattern, pass 3 shows it.
4. Diffuse+decay pass runs; the test pattern visibly blurs and fades.
5. Agents exist, deposit only, no sensing — you get a starfield that smears.
6. Sensing and steering. This is the step where it becomes a slime mold.
7. egui panel wired to `params`.

Each step ends with something on screen. If a step goes dark, the bug is in
that step. Do not build 1–6 and then debug.
