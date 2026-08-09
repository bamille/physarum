// Pass 3 — agent billboards.
//
// Not written yet. `Renderer::build_pipeline` looks for `@vertex` and
// `@fragment` in this file and stays in clear-only mode until both exist, so
// the app runs (black window) with this file as-is.
//
// The Rust side has already committed to the following interface. Match it.
//
// -- draw call ---------------------------------------------------------------
//
//   draw(0..4, 0..n_agents)      topology = triangle-strip, no vertex buffers
//
//   @builtin(vertex_index)   0..4  -> which corner of the quad
//   @builtin(instance_index) 0..n  -> which agent
//
//   Strip corner order for vertex_index i, as (x, y) in [-1, 1]:
//       0 -> (-1, -1)   1 -> (-1, +1)   2 -> (+1, -1)   3 -> (+1, +1)
//   i.e. x = f32(i >> 1u) * 2.0 - 1.0, y = f32(i & 1u) * 2.0 - 1.0
//
// -- bindings, group(0) ------------------------------------------------------
//
//   @group(0) @binding(0) var<uniform> camera: Camera;         // VERTEX | FRAGMENT
//   @group(0) @binding(1) var<storage, read> agents: array<Agent>;  // VERTEX
//
//   struct Camera {              // src/lib.rs `Camera`, same order
//       view_proj: mat4x4<f32>,  // proj * view, world -> clip
//       right:     vec4<f32>,    // world-space camera basis, w = 0
//       up:        vec4<f32>,
//       params:    vec4<f32>,    // x = agent radius (world units)
//                                // y = speed_scale, z/w free
//   };
//
//   struct Agent {               // src/lib.rs `Agent`, 24 bytes
//       pos:     vec2<f32>,      // world XY; agents live on the z = 0 plane
//       heading: vec2<f32>,      // unit vector
//       speed:   f32,
//       _pad:    f32,
//   };
//
// -- vs_main -----------------------------------------------------------------
//
//   center     = vec3(agents[instance_index].pos, 0.0)
//   world      = center + camera.right.xyz * corner.x * camera.params.x
//                       + camera.up.xyz    * corner.y * camera.params.x
//   position   = camera.view_proj * vec4(world, 1.0)
//
//   Pass `corner` through to the fragment stage as the quad-local coordinate.
//
// -- fs_main -----------------------------------------------------------------
//
//   Blending is additive with SrcAlpha as the source factor, and the clear is
//   near-black, so alpha is the brightness knob: discard or fade to a =: 0
//   outside length(corner) > 1.0 to get a round dot instead of a square.
