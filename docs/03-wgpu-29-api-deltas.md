# wgpu 29 / egui 0.35 / glam 0.33 — what changed

Offline, you cannot search for why a snippet fails to compile. wgpu 27→29 moved
a great deal, so essentially every wgpu tutorial, blog post, and Stack Overflow
answer you have in your head or in a saved tab is written against an older API
and **will not compile**. These are the ones you will actually hit.

## wgpu

| Older tutorials say | wgpu 29 wants |
|---|---|
| `Instance::new(&InstanceDescriptor { .. })` | `Instance::new(InstanceDescriptor::new_without_display_handle())` |
| `bind_group_layouts: &[&bgl]` | `bind_group_layouts: &[Some(&bgl)]` |
| `push_constant_ranges: &[]` | `immediate_size: 0` — push constants were renamed "immediates" |
| `depth_write_enabled: true` | `Some(true)`, and `depth_compare: Some(..)` |
| `device.poll(Maintain::Wait)` | `device.poll(PollType::wait_indefinitely())?` |
| `surface.get_current_texture() -> Result<..>` | returns the `CurrentSurfaceTexture` enum |
| `RenderPassDescriptor { .. }` | gained `multiview_mask` |
| `RenderPassColorAttachment { .. }` | gained `depth_slice` |
| `SamplerDescriptor { mipmap_filter: FilterMode }` | `MipmapFilterMode` |

## egui 0.35

| Older | 0.35 |
|---|---|
| `Context::run(input, \|ctx\| ..)` | `Context::run_ui(input, \|ui\| ..)` |
| `egui::SidePanel::left(id).show(ctx, ..)` | `egui::Panel::left(id).show(ui, ..)` |
| `Line::new(points)` | `Line::new("name", points)` |

## glam 0.33

`Mat4::perspective_rh` and `Mat4::look_at_rh` are **deprecated**:

```rust
glam::camera::rh::proj::directx::perspective(fovy, aspect, near, far)  // z in 0..1
glam::camera::rh::view::look_at_mat4(eye, target, up)
```

The `directx` module is documented as "for use with DirectX and WebGPU". Using
it means you never need the `OPENGL_TO_WGPU_MATRIX` fudge that older wgpu
tutorials carry around. (Mostly irrelevant for a 2D slime mold, but you will
want it the moment you add a camera.)

---

## Backend notes for this machine

This is macOS, so wgpu runs on **Metal**. The course was verified on Windows /
Vulkan / RTX 2060, and a few things differ:

- `WGPU_BACKEND=metal` is the only real option. `WGPU_BACKEND=gl` exists and is
  useless for compute.
- Metal's workgroup (threadgroup) memory limit is 32 KiB on Apple silicon,
  compared to the 48 KiB the course quotes for NVIDIA. If you port a workgroup
  tiling kernel from the course, check `max_compute_workgroup_storage_size` at
  runtime rather than assuming.
- Timestamp queries work on Metal but the feature must be requested
  (`Features::TIMESTAMP_QUERY`); on some drivers you also want
  `TIMESTAMP_QUERY_INSIDE_ENCODERS` to time individual passes.
- There is no RenderDoc. Xcode's **Metal frame capture** is the equivalent, and
  it is good — it will show you buffer contents and dispatch arguments. It
  needs the app launched from Xcode or with
  `METAL_CAPTURE_ENABLED=1` plus a programmatic capture scope.

## Debugging levers that cost nothing

- `RUST_LOG=wgpu_core=warn` turns on wgpu's validation chatter. `RUST_LOG=trace`
  is overwhelming but is sometimes the only way to learn which resource a
  validation error refers to.
- **Label everything.** Every descriptor takes `label: Some("...")`. Validation
  errors quote the label. Unlabelled resources produce errors about
  `<Buffer-(0, 3, Metal)>`, which tells you nothing at 30,000 feet.
- `device.on_uncaptured_error(..)` to fail loudly instead of getting a silently
  broken frame.
