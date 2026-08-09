# Working offline

Everything needed to build this project is on disk. Here is exactly what works,
what does not, and how to repair it if you break it.

## How it is set up

- `Cargo.lock` pins all 383 transitive crates to exact versions.
- `vendor/` holds the complete source of those 383 crates (565 MB).
- `.cargo/config.toml` replaces the crates.io source with `vendor/`, so cargo
  reads from disk and never opens a socket. It also sets `net.offline = true`,
  which turns any accidental fetch into an immediate error rather than a
  30-second timeout.
- `target/` is fully warmed — a clean `cargo build` has already run, so
  incremental builds are fast from the first minute.

This means `~/.cargo/registry` is irrelevant to this project. You can wipe it
and this still builds.

## Works offline

```bash
cargo build
cargo run
cargo test
cargo check
cargo clippy
cargo doc --open
cargo tree
```

## Does not work offline

| Command | Why | What to do |
|---|---|---|
| `cargo add <crate>` | needs the registry index | add every dependency you might want **before** you leave |
| `cargo update` | same | don't |
| `cargo generate-lockfile` after editing deps | same | same |
| `rustup update` / `rustup component add` | downloads toolchains | install `clippy`, `rustfmt`, `rust-src`, `rust-analyzer` now |

## Offline documentation

Full rustdoc for the entire dependency tree — wgpu, winit, egui, glam,
bytemuck, everything — is generated at `target/doc/`.

```bash
cargo doc --open
```

Or open `target/doc/wgpu/index.html` directly. The search box works offline;
it is a local JS index, not a network call.

**Do not run `cargo clean`.** It deletes `target/`, which takes both the warm
build cache and all the generated documentation with it. Regenerating needs no
network (`cargo doc` works offline), but it takes several minutes of battery.
If you want the docs somewhere safe from `cargo clean`:

```bash
cp -r target/doc ~/rust-docs-airplane
```

The WGSL spec is not on disk and is not a crate. If you want it, save
`https://www.w3.org/TR/WGSL/` as a single-file HTML page before you go — it is
the one reference you will miss most, particularly the built-in function list.

## Repairing the vendor directory

If you need to change dependencies, you need network. While online:

1. Comment out the `[net] offline = true` block in `.cargo/config.toml`.
2. Edit `Cargo.toml`. Keep pins exact (`=x.y.z`) — a caret range that later
   resolves to a version not present in `vendor/` fails with a confusing
   "no matching package" error rather than an obvious one.
3. ```bash
   cargo generate-lockfile
   cargo vendor --versioned-dirs
   cargo build
   cargo doc
   ```
4. Restore the `offline = true` block.

`cargo vendor` prints the config stanza it wants appended; it is already in
`.cargo/config.toml`, so ignore that output.

## Sanity check before you disconnect

```bash
cargo build && cargo test && test -f target/doc/wgpu/index.html && echo OK
```

If that prints `OK`, you are self-sufficient.
