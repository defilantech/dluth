//! The attention sublayer's operations on the GPU against the CPU reference functions.
//! GPU tests; they need an NVIDIA GPU of the architecture the kernels were built for.

use dluth_core::{Backend, Buffer};
use dluth_cuda::attention::{Attention, Heads, History, Relative};
use dluth_cuda::floats::Floats;
use dluth_cuda::{CudaDevice, CudaError, Module};
use dluth_kernels::{cpu, sconv, sinkattn};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, size: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * size).collect()
    }
}

fn put(dev: &CudaDevice, buf: &Buffer, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    dev.upload(buf, &bytes).unwrap();
}

fn upload(dev: &CudaDevice, values: &[f32]) -> Buffer {
    let buf = dev.alloc(values.len().max(1) * 4).unwrap();
    put(dev, &buf, values);
    buf
}

fn download(dev: &CudaDevice, buf: &Buffer, n: usize) -> Vec<f32> {
    dev.synchronize().unwrap();
    let mut bytes = vec![0u8; n * 4];
    dev.download(buf, &mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// Every element within `ulps` roundings of the reference, relative to `size` (the sum of
/// the sizes of the terms that made it, or the value itself for a product of factors).
fn close(what: &str, got: &[f32], want: &[f32], size: impl Fn(usize) -> f32, ulps: f32) {
    assert_eq!(got.len(), want.len(), "{what}");
    for i in 0..got.len() {
        let bound = ulps * f32::EPSILON * size(i).max(f32::MIN_POSITIVE);
        assert!((got[i] - want[i]).abs() <= bound, "{what}: element {i}: {} vs {} (bound {bound})", got[i], want[i]);
    }
}

fn kernels(dev: &CudaDevice) -> (Module<'_>,) {
    (Module::load_embedded(dev).unwrap(),)
}

#[test]
fn scaling_clipping_and_the_late_factor_match_the_cpu() {
    let dev = CudaDevice::open(0).expect("GPU");
    let (module,) = kernels(&dev);
    let at = Attention::new(&dev, &module).unwrap();
    let mut rng = Lcg(1);
    // Real head shapes (24 heads of 128; 24 of 16) and an odd one. Vectors of very different
    // lengths, so that some are over the clip and some under; one all zeros.
    for (heads, dim, clip) in [(24, 128, 22.63f32), (24, 16, 8.0), (5, 3, 1.0)] {
        let mut x: Vec<f32> = Vec::new();
        for h in 0..heads {
            let size = [0.0, 0.01, 1.0, 5.0, 40.0][h % 5];
            x.extend(rng.vec(dim, size));
        }
        let scale = rng.vec(heads * dim, 1.5);
        for (with_scale, late) in [(true, 1.0718f32), (false, 1.0)] {
            let (dx, ds) = (upload(&dev, &x), upload(&dev, &scale));
            let s = with_scale.then(|| Floats::first(&ds, heads * dim));
            at.head_scale_clip(Floats::first(&dx, heads * dim), s, heads, dim, clip, late).unwrap();
            let got = download(&dev, &dx, heads * dim);
            let mut want = x.clone();
            if with_scale {
                want.iter_mut().zip(&scale).for_each(|(v, s)| *v *= s);
            }
            for head in want.chunks_exact_mut(dim) {
                cpu::l2_clip_inplace(head, clip);
                head.iter_mut().for_each(|v| *v *= late);
            }
            // A product of up to four factors, one of them from a sum of squares that the
            // device may round differently in its last bits.
            close("head_scale_clip", &got, &want, |i| want[i].abs(), 8.0);
            let lengths: Vec<f32> =
                want.chunks_exact(dim).map(|h| h.iter().map(|v| v * v).sum::<f32>().sqrt()).collect();
            assert!(lengths.iter().any(|l| (l / late - clip).abs() < 1e-3 * clip), "no head was clipped: {lengths:?}");
            assert!(lengths.iter().any(|l| *l > 0.0 && l / late < 0.5 * clip), "no head was left alone");
            dev.free(dx).unwrap();
            dev.free(ds).unwrap();
        }
    }
}

#[test]
fn value_normalization_matches_the_cpu() {
    let dev = CudaDevice::open(0).expect("GPU");
    let (module,) = kernels(&dev);
    let at = Attention::new(&dev, &module).unwrap();
    let mut rng = Lcg(2);
    for (heads, dim) in [(8, 128), (3, 5), (1, 1)] {
        let mut x = rng.vec(heads * dim, 3.0);
        x[..dim].fill(0.0); // a head of zeros: only eps keeps it finite
        let (dx, dout) = (upload(&dev, &x), dev.alloc(heads * dim * 4).unwrap());
        at.head_rmsnorm_plain(Floats::first(&dx, heads * dim), Floats::first(&dout, heads * dim), heads, dim, 1e-6)
            .unwrap();
        let got = download(&dev, &dout, heads * dim);
        let mut want = vec![0.0f32; heads * dim];
        for (o, head) in want.chunks_exact_mut(dim).zip(x.chunks_exact(dim)) {
            cpu::rmsnorm_plain(head, 1e-6, o);
        }
        close("head_rmsnorm_plain", &got, &want, |i| want[i].abs(), 8.0);
        assert!(got[..dim].iter().all(|v| *v == 0.0));
    }
}

/// Slides the CPU reference's memory of recent tokens forward by one, oldest first.
fn remember(recent: &mut [f32], current: &[f32]) {
    let n = current.len();
    if recent.len() >= n {
        recent.copy_within(n.., 0);
        let at = recent.len() - n;
        recent[at..].copy_from_slice(current);
    }
}

#[test]
fn both_short_convolutions_match_the_cpu_token_after_token() {
    let dev = CudaDevice::open(0).expect("GPU");
    let (module,) = kernels(&dev);
    let at = Attention::new(&dev, &module).unwrap();
    let mut rng = Lcg(3);
    // One tap (no memory at all), two, and the model's four; channels not a multiple of anything.
    for (channels, n, heads) in [(36, 4, 4), (10, 2, 5), (7, 1, 7), (1024, 4, 8)] {
        let fixed_taps = rng.vec(channels * n, 0.7);
        let (d_taps, d_cur, d_out) =
            (upload(&dev, &fixed_taps), dev.alloc(channels * 4).unwrap(), dev.alloc(channels * 4).unwrap());
        let d_token_taps = dev.alloc(heads * n * 4).unwrap();
        // Two rings, one per kernel, never zeroed: a ring's old contents must not leak into
        // the first tokens of a sequence.
        let junk = vec![99.0f32; (n - 1).max(1) * channels];
        let (ring_fixed, ring_dynamic) = (upload(&dev, &junk), upload(&dev, &junk));
        let ring = |b| Floats::first(b, (n - 1) * channels);
        let (mut recent_fixed, mut recent_dynamic) =
            (vec![0.0f32; (n - 1) * channels], vec![0.0f32; (n - 1) * channels]);
        for pos in 0..11 {
            let current = rng.vec(channels, 2.0);
            let token_taps = rng.vec(heads * n, 0.5);
            put(&dev, &d_cur, &current);
            put(&dev, &d_token_taps, &token_taps);
            let cur = Floats::first(&d_cur, channels);

            at.sconv_fixed(
                cur,
                ring(&ring_fixed),
                Floats::first(&d_taps, channels * n),
                Floats::first(&d_out, channels),
                n,
                pos,
            )
            .unwrap();
            let got = download(&dev, &d_out, channels);
            let mut want = vec![0.0f32; channels];
            sconv::fixed(&recent_fixed, &current, &fixed_taps, &mut want);
            // n + 1 terms added in the reference's order; the device may fuse each multiply
            // into its add. The terms are of size about 1.
            close(&format!("sconv_fixed, {n} taps, token {pos}"), &got, &want, |_| 8.0, 4.0);
            remember(&mut recent_fixed, &current);

            at.sconv_dynamic(
                cur,
                ring(&ring_dynamic),
                Floats::first(&d_taps, channels * n),
                Floats::first(&d_token_taps, heads * n),
                Floats::first(&d_out, channels),
                n,
                heads,
                pos,
            )
            .unwrap();
            let got = download(&dev, &d_out, channels);
            sconv::dynamic(&recent_dynamic, &current, &fixed_taps, &token_taps, heads, &mut want);
            close(&format!("sconv_dynamic, {n} taps, token {pos}"), &got, &want, |_| 8.0, 4.0);
            remember(&mut recent_dynamic, &current);
        }
    }
}

/// Runs `tokens` tokens through attention with a history of `cap` rows on the device and
/// compares every output with `sinkattn::attend` over the tokens a window of `window` shows.
fn attend_case(
    dev: &CudaDevice,
    at: &Attention,
    shape: Heads,
    relative: (usize, usize),
    cap: usize,
    window: usize,
    tokens: usize,
) {
    let (rel_dim, reach) = relative;
    let Heads { heads, kv_heads, dim } = shape;
    let width = kv_heads * dim;
    let mut rng = Lcg((heads * 131 + cap * 7 + reach) as u64);
    let table = rng.vec(rel_dim * reach, 0.8);
    let d_table = upload(dev, &table);
    let (d_keys, d_values) = (upload(dev, &vec![7.0f32; cap * width]), upload(dev, &vec![7.0f32; cap * width]));
    let (d_q, d_rel, d_sinks) =
        (dev.alloc(heads * dim * 4).unwrap(), dev.alloc(heads * rel_dim * 4).unwrap(), dev.alloc(heads * 4).unwrap());
    let (d_weights, d_out) = (dev.alloc(heads * (cap + 1) * 4).unwrap(), dev.alloc(heads * dim * 4).unwrap());
    let (mut host_keys, mut host_values) = (vec![7.0f32; cap * width], vec![7.0f32; cap * width]);
    let (mut all_keys, mut all_values): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    for pos in 0..tokens {
        let (key, value) = (rng.vec(width, 1.0), rng.vec(width, 2.0));
        let (q, rel, sinks) = (rng.vec(heads * dim, 1.5), rng.vec(heads * rel_dim, 1.0), rng.vec(heads, 3.0));
        // The token's key and value go into row pos % cap, as the model does it.
        let row = (pos % cap) * width;
        host_keys[row..row + width].copy_from_slice(&key);
        host_values[row..row + width].copy_from_slice(&value);
        put(dev, &d_keys, &host_keys);
        put(dev, &d_values, &host_values);
        put(dev, &d_q, &q);
        put(dev, &d_rel, &rel);
        put(dev, &d_sinks, &sinks);
        all_keys.extend_from_slice(&key);
        all_values.extend_from_slice(&value);
        let seen = (pos + 1).min(window);

        at.attend(
            shape,
            Floats::first(&d_q, heads * dim),
            History { keys: Floats::first(&d_keys, cap * width), values: Floats::first(&d_values, cap * width), cap },
            Relative {
                rel: Floats::first(&d_rel, heads * rel_dim),
                table: Floats::first(&d_table, rel_dim * reach),
                rel_dim,
                reach,
            },
            Floats::first(&d_sinks, heads),
            Floats::first(&d_weights, heads * (seen + 1)),
            Floats::first(&d_out, heads * dim),
            pos,
            seen,
        )
        .unwrap();
        let got = download(dev, &d_out, heads * dim);

        // The reference: the seen tokens oldest first, and each head's bias by distance.
        let first = pos + 1 - seen;
        let (keys, values) = (&all_keys[first * width..], &all_values[first * width..]);
        let near = seen.min(reach);
        let mut bias = vec![0.0f32; heads * seen];
        let mut profile = vec![0.0f32; near];
        for (head, row) in rel.chunks_exact(rel_dim).zip(bias.chunks_exact_mut(seen)) {
            sinkattn::distance_profile(head, &table, reach, &mut profile);
            for (distance, &p) in profile.iter().enumerate() {
                row[seen - 1 - distance] = p;
            }
        }
        let mut want = vec![0.0f32; heads * dim];
        sinkattn::attend(
            sinkattn::Query {
                q: &q,
                keys,
                values,
                bias: &bias,
                sinks: &sinks,
                heads,
                kv_heads,
                head_dim: dim,
                positions: seen,
            },
            &mut want,
        );
        // Weights sum to at most 1 and values are below 2 in size. The softmax's sum is
        // added in a different order on the device, which moves every weight by a few
        // roundings of 1; the scores feeding it are sums of `dim` products of size about 1.
        let bound = 64.0 * (dim as f32) * f32::EPSILON;
        for i in 0..want.len() {
            assert!(
                (got[i] - want[i]).abs() <= bound,
                "{heads} heads, cap {cap}, window {window}, reach {reach}, token {pos}: element {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
fn attention_matches_the_cpu_with_and_without_a_window() {
    let dev = CudaDevice::open(0).expect("GPU");
    let (module,) = kernels(&dev);
    let at = Attention::new(&dev, &module).unwrap();
    let shape = Heads { heads: 6, kv_heads: 2, dim: 8 };
    // A layer that sees everything: the history is as long as the sequence. Reach shorter
    // than the sequence, so far tokens get no relative-position term; and longer.
    attend_case(&dev, &at, shape, (4, 5), 32, usize::MAX, 20);
    attend_case(&dev, &at, shape, (4, 40), 20, usize::MAX, 20);
    // A layer with a window of 7: the history wraps around, three times over.
    attend_case(&dev, &at, shape, (4, 7), 7, 7, 23);
    // One head, one position of history: every token sees only itself.
    attend_case(&dev, &at, Heads { heads: 1, kv_heads: 1, dim: 4 }, (2, 3), 1, 1, 4);
    // The real shape of a head, at a few hundred tokens.
    attend_case(&dev, &at, Heads { heads: 24, kv_heads: 8, dim: 128 }, (16, 1024), 300, usize::MAX, 300);
}

#[test]
fn the_output_gate_matches_the_cpu() {
    let dev = CudaDevice::open(0).expect("GPU");
    let (module,) = kernels(&dev);
    let at = Attention::new(&dev, &module).unwrap();
    let mut rng = Lcg(5);
    let n = 3073;
    let (x, gate) = (rng.vec(n, 4.0), rng.vec(n, 30.0));
    let (dx, dg) = (upload(&dev, &x), upload(&dev, &gate));
    at.sigmoid_mul(Floats::first(&dx, n), Floats::first(&dg, n)).unwrap();
    let got = download(&dev, &dx, n);
    let want: Vec<f32> = x.iter().zip(&gate).map(|(x, g)| x * cpu::sigmoid(*g)).collect();
    close("sigmoid_mul", &got, &want, |i| want[i].abs(), 8.0);
}

#[test]
fn requests_that_would_leave_the_buffers_are_refused() {
    let dev = CudaDevice::open(0).expect("GPU");
    let (module,) = kernels(&dev);
    let at = Attention::new(&dev, &module).unwrap();
    let a = dev.alloc(64 * 4).unwrap();
    let b = dev.alloc(64 * 4).unwrap();
    let c = dev.alloc(1024 * 4).unwrap();
    let f = Floats::first;
    let refused = |what: &str, r: Result<(), CudaError>| {
        assert!(matches!(r, Err(CudaError::Refused(_))), "{what} was not refused")
    };

    refused("more heads than the buffer holds", at.head_scale_clip(f(&a, 80), None, 10, 8, 1.0, 1.0));
    refused(
        "a run that starts past its buffer",
        at.head_scale_clip(Floats { buffer: &a, at: 60, len: 8 }, None, 1, 8, 1.0, 1.0),
    );
    refused("a clip of zero", at.head_scale_clip(f(&a, 64), None, 8, 8, 0.0, 1.0));
    refused("a clip that is not a number", at.head_scale_clip(f(&a, 64), None, 8, 8, f32::NAN, 1.0));
    refused("a scale that is the vectors", at.head_scale_clip(f(&a, 64), Some(f(&a, 64)), 8, 8, 1.0, 1.0));
    refused("normalizing in place", at.head_rmsnorm_plain(f(&a, 64), f(&a, 64), 8, 8, 1e-6));
    refused("a negative eps", at.head_rmsnorm_plain(f(&a, 64), f(&b, 64), 8, 8, -1.0));

    refused("a ring too short for the taps", at.sconv_fixed(f(&a, 16), f(&b, 32), f(&c, 64), f(&c, 16), 4, 0));
    refused(
        "taps too few",
        at.sconv_fixed(f(&a, 16), f(&b, 48), f(&c, 48), Floats { buffer: &c, at: 100, len: 16 }, 4, 0),
    );
    refused("the output over the token", at.sconv_fixed(f(&a, 16), f(&b, 48), f(&c, 64), f(&a, 16), 4, 0));
    refused("no taps", at.sconv_fixed(f(&a, 16), f(&b, 48), f(&c, 64), Floats { buffer: &c, at: 100, len: 16 }, 0, 0));
    refused(
        "heads that do not divide the channels",
        at.sconv_dynamic(
            f(&a, 16),
            f(&b, 48),
            f(&c, 64),
            Floats { buffer: &c, at: 64, len: 12 },
            Floats { buffer: &c, at: 100, len: 16 },
            4,
            3,
            0,
        ),
    );

    let shape = Heads { heads: 4, kv_heads: 2, dim: 4 };
    let history = |cap| History { keys: f(&a, cap * 8), values: f(&b, cap * 8), cap };
    let relative = Relative {
        rel: Floats { buffer: &c, at: 0, len: 8 },
        table: Floats { buffer: &c, at: 8, len: 6 },
        rel_dim: 2,
        reach: 3,
    };
    let (q, sinks, out) = (
        Floats { buffer: &c, at: 16, len: 16 },
        Floats { buffer: &c, at: 32, len: 4 },
        Floats { buffer: &c, at: 40, len: 16 },
    );
    let weights = |seen: usize| Floats { buffer: &c, at: 64, len: 4 * (seen + 1) };
    at.attend(shape, q, history(8), relative, sinks, weights(3), out, 2, 3).unwrap();
    dev.synchronize().unwrap();
    refused(
        "more tokens seen than have been fed",
        at.attend(shape, q, history(8), relative, sinks, weights(4), out, 2, 4),
    );
    refused(
        "more tokens seen than the history holds",
        at.attend(shape, q, history(8), relative, sinks, weights(9), out, 20, 9),
    );
    refused("no token seen", at.attend(shape, q, history(8), relative, sinks, weights(0), out, 2, 0));
    refused(
        "a history larger than its buffers",
        at.attend(shape, q, history(9), relative, sinks, weights(3), out, 2, 3),
    );
    refused("scratch of the wrong size", at.attend(shape, q, history(8), relative, sinks, weights(2), out, 2, 3));
    refused("the output over the query", at.attend(shape, q, history(8), relative, sinks, weights(3), q, 2, 3));
    refused(
        "heads that the key-value heads do not divide",
        at.attend(Heads { heads: 3, kv_heads: 2, dim: 4 }, q, history(8), relative, sinks, weights(3), out, 2, 3),
    );
    refused("a gate that is the values", at.sigmoid_mul(f(&a, 64), f(&a, 64)));
    refused("a gate shorter than the values", at.sigmoid_mul(f(&a, 64), f(&b, 60)));
}
