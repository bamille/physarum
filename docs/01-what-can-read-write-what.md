# What can read and write what

The reference you asked for. Everything below is wgpu 29 / WGSL as pinned in
`Cargo.toml`. When in doubt, the local rustdoc (`cargo doc --open`, see
[04-offline-workflow.md](04-offline-workflow.md)) is authoritative and offline.

---

## 1. Address spaces

Every variable in WGSL lives in an address space, and the address space decides
who can see it and whether it is writable.

| Space | Declaration | Access | Lifetime / scope | Notes |
|---|---|---|---|---|
| `function` | `var x = 0.0;` `let y = ...;` | read_write | one invocation, one call | registers |
| `private` | `var<private> x: f32;` | read_write | one invocation, whole shader | module scope |
| `workgroup` | `var<workgroup> tile: array<f32, 64>;` | read_write | one workgroup, one dispatch | **compute only**, on-chip, ~32 KiB budget |
| `uniform` | `var<uniform> p: Params;` | **read only** | whole dispatch/draw | ≤64 KiB, strict 16-byte layout |
| `storage` | `var<storage, read>` / `var<storage, read_write>` | as declared | whole dispatch/draw | the workhorse |
| `handle` | `var t: texture_2d<f32>;` `var s: sampler;` | opaque | whole dispatch/draw | textures and samplers |

Two rules people trip over:

- **`workgroup` is compute-only.** There is no shared memory in a vertex or
  fragment shader.
- **`uniform` is read-only, always.** There is no `var<uniform, read_write>`.
  If you want to write it, it is a storage buffer.

---

## 2. Buffers

| WGSL declaration | `wgpu::BindingType` | Required `BufferUsages` | Size cap | Legal shader stages |
|---|---|---|---|---|
| `var<uniform> p: P;` | `Buffer { ty: Uniform }` | `UNIFORM` | 64 KiB (`max_uniform_buffer_binding_size`) | vertex, fragment, compute |
| `var<storage, read> a: array<T>;` | `Buffer { ty: Storage { read_only: true } }` | `STORAGE` | `max_storage_buffer_binding_size` (128 MiB+) | vertex, fragment, compute |
| `var<storage, read_write> a: array<T>;` | `Buffer { ty: Storage { read_only: false } }` | `STORAGE` | same | **fragment, compute — not vertex** |

**A vertex shader cannot write.** A bind group layout that declares a
read-write storage buffer with `ShaderStages::VERTEX` in its visibility is a
validation error, not a runtime surprise. Fragment shaders *can* write to
storage buffers, though you rarely want to — you have no control over how many
times a fragment runs.

### Usage flags that conflict

- `MAP_READ` may only be combined with `COPY_DST`.
- `MAP_WRITE` may only be combined with `COPY_SRC`.

So a buffer you read back on the CPU is a *separate* staging buffer
(`MAP_READ | COPY_DST`) that you `copy_buffer_to_buffer` into. You cannot map
the storage buffer the shader wrote to. This is the single most common "why
won't this compile / why is this a validation error" moment in readback code.

### Layout: uniform vs storage

`uniform` rounds array element stride up to a multiple of 16 bytes. `storage`
does not. That means:

```wgsl
struct Agent { pos: vec2<f32>, angle: f32 }   // 12 bytes
var<storage, read_write> agents: array<Agent>;  // stride 12 — fine
var<uniform>             agents: array<Agent, 64>; // stride 16 — silently different!
```

Also: `vec3<f32>` has **size 12 but alignment 16**. A `vec3` followed by an
`f32` packs into 16 bytes; a `vec3` followed by another `vec3` does not. If a
field is mysteriously reading as garbage, it is almost always this. Use `vec4`
or add explicit padding fields, mirror the struct in Rust with `#[repr(C)]` +
`bytemuck::Pod`, and assert `size_of::<T>()` in a test.

### Atomics

`atomic<u32>` and `atomic<i32>` are legal **only** in `storage` (declared
`read_write`) and `workgroup`. Not in `uniform`, not in `private`, not in
textures.

**There is no `atomic<f32>`.** For a slime mold trail this matters — see
[02-slime-mold-architecture.md](02-slime-mold-architecture.md). Your options are
a fixed-point `atomic<u32>`, a compare-exchange loop, or simply accepting lost
deposits (which for this sim is visually invisible).

---

## 3. Textures

| WGSL declaration | `wgpu::BindingType` | Required `TextureUsages` | Read | Write | Filtering |
|---|---|---|---|---|---|
| `texture_2d<f32>` | `Texture { sample_type: Float { filterable: true } }` | `TEXTURE_BINDING` | `textureSample` / `textureSampleLevel` / `textureLoad` | ✗ | ✓ (needs a `sampler`) |
| `texture_storage_2d<F, write>` | `StorageTexture { access: WriteOnly }` | `STORAGE_BINDING` | ✗ | `textureStore` | ✗ |
| `texture_storage_2d<F, read>` | `StorageTexture { access: ReadOnly }` | `STORAGE_BINDING` | `textureLoad` | ✗ | ✗ |
| `texture_storage_2d<F, read_write>` | `StorageTexture { access: ReadWrite }` | `STORAGE_BINDING` | `textureLoad` | `textureStore` | ✗ |

Rules attached to that table:

- **Storage textures are never filtered and never sampled.** No `sampler`, no
  bilinear interpolation, no mips. `textureLoad` with integer coordinates only.
  If you want smooth reads, bind the texture a second way — see below.
- **`read_write` access is format-restricted** to the single-channel 32-bit
  formats (`r32float`, `r32uint`, `r32sint`). Do not plan a design around a
  `read_write` `rgba16float`. Check `wgpu::StorageTextureAccess` in the local
  rustdoc before committing to it.
- **Not every format can be a storage texture.** The core set is `rgba8unorm`,
  `rgba8snorm`, `rgba8uint`, `rgba8sint`, `rgba16uint/sint/float`,
  `r32uint/sint/float`, `rg32*`, `rgba32*`. Notably **no sRGB formats** and no
  `bgra8unorm` without the `BGRA8UNORM_STORAGE` feature.
- **`textureSample` is fragment-only.** It needs implicit derivatives, which
  only exist in a fragment quad. In compute, use `textureSampleLevel(t, s, uv,
  0.0)`.
- **No atomics on textures** in core. wgpu exposes
  `StorageTextureAccess::Atomic` behind the `TEXTURE_ATOMIC` feature
  (`r32uint`); assume you do not have it unless you check.

### The one texture, two bindings trick

A texture created with `TEXTURE_BINDING | STORAGE_BINDING` can be written by a
compute pass through a storage binding, then sampled by a later render pass
through a sampled binding. That is the normal way to get a compute-generated
field onto the screen with smooth filtering.

What you **cannot** do is bind the same subresource as a writable storage
texture *and* as anything else **within the same pass**. WebGPU's usage-scope
validation rejects it. Different passes, fine. Same pass, error.

### Copies

`copy_texture_to_buffer` requires `bytes_per_row` to be a multiple of **256**.
A 100×100 `rgba8` texture is 400 bytes per row, so you must pad the staging
buffer to 512 and strip the padding on the CPU. Forgetting this produces an
image that shears diagonally — an instantly recognisable symptom once you have
seen it once.

---

## 4. Who is ordered relative to whom

This is the part that produces bugs that look like physics.

| Situation | Ordered? |
|---|---|
| Two invocations in the same workgroup, separated by `workgroupBarrier()` | ✓ for `workgroup` memory |
| Two invocations in the same workgroup, separated by `storageBarrier()` | ✓ for `storage` memory |
| Two invocations in **different workgroups**, same dispatch | ✗ **nothing orders them** |
| Two consecutive `dispatch_workgroups` calls | ✓ WebGPU makes dispatches behave as if fully ordered, with writes visible to the next |
| Two passes in the same encoder | ✓ |

The middle row is the whole reason ping-pong exists. `new[i] = f(old[i-1],
old[i], old[i+1])` done in place is a race across workgroups, and it does not
crash and does not trip validation. It produces a field that is *mostly* right
with a scatter of cells that read a half-updated neighbour. Read `a`, write
`b`, swap.

The fourth row is the good news: you do **not** need manual barriers between
dispatches. Encode agent-update, then diffuse, then render, and the ordering is
handled for you.

---

## 5. Quick decision table

| I want to… | Use |
|---|---|
| Pass 8 tuning parameters, updated per frame | uniform buffer |
| Hold 1M agents, updated in place | `var<storage, read_write> array<Agent>` |
| Hold a 2D field the shader reads neighbours from | ping-ponged storage buffers, or two storage textures |
| Accumulate scattered deposits from many agents | `array<atomic<u32>>` storage buffer, fixed-point |
| Show a field on screen with smooth filtering | storage texture written in compute, sampled in fragment |
| Show a field on screen without filtering | read the storage buffer directly from the fragment shader |
| Get numbers back to the CPU | `COPY_SRC` storage buffer → `copy_buffer_to_buffer` → `MAP_READ \| COPY_DST` staging buffer |
