//! The small vector operations on the GPU against the CPU reference functions.
//! GPU tests; they need an NVIDIA GPU of the architecture the kernels were built for.

use dluth_core::{Backend, Buffer};
use dluth_cuda::floats::Floats;
use dluth_cuda::small::Small;
use dluth_cuda::{CudaDevice, CudaError, Module};
use dluth_kernels::{cpu, depthmix};

struct Lcg(u64);
impl Lcg {
    fn vec(&mut self, n: usize, size: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * size
            })
            .collect()
    }
}

fn upload(dev: &CudaDevice, values: &[f32]) -> Buffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = dev.alloc(bytes.len()).unwrap();
    dev.upload(&buf, &bytes).unwrap();
    buf
}

fn download(dev: &CudaDevice, buf: &Buffer, n: usize) -> Vec<f32> {
    dev.synchronize().unwrap();
    let mut bytes = vec![0u8; n * 4];
    dev.download(buf, &mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

#[test]
fn add_is_exact() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let small = Small::new(&dev, &module).unwrap();
    let mut rng = Lcg(1);
    let n = 2305;
    let (a, b) = (rng.vec(n, 10.0), rng.vec(n, 0.001));
    let (da, db, dout) = (upload(&dev, &a), upload(&dev, &b), dev.alloc(n * 4).unwrap());
    small.add(Floats::first(&da, n), Floats::first(&db, n), Floats::first(&dout, n)).unwrap();
    let got = download(&dev, &dout, n);
    // One addition per element: no room for a difference.
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(got, want);
}

#[test]
fn weighted_normalization_matches_the_cpu() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let small = Small::new(&dev, &module).unwrap();
    let mut rng = Lcg(2);
    // One vector, and batches of them sharing the weights; a vector of zeros among them.
    for (n, batch) in [(1, 1), (7, 3), (256, 1), (2304, 5)] {
        let (mut x, w) = (rng.vec(batch * n, 5.0), rng.vec(n, 2.0));
        if batch > 1 {
            x[n..2 * n].fill(0.0);
        }
        let (dx, dw, dout) = (upload(&dev, &x), upload(&dev, &w), dev.alloc(batch * n * 4).unwrap());
        small
            .rmsnorm(Floats::first(&dx, batch * n), Floats::first(&dw, n), Floats::first(&dout, batch * n), 1e-6)
            .unwrap();
        let got = download(&dev, &dout, batch * n);
        let mut want = vec![0.0f32; batch * n];
        for (o, v) in want.chunks_exact_mut(n).zip(x.chunks_exact(n)) {
            cpu::rmsnorm(v, &w, 1e-6, o);
        }
        // The squares are added in another order, which moves the scale by a few roundings.
        for i in 0..batch * n {
            assert!(
                (got[i] - want[i]).abs() <= 16.0 * f32::EPSILON * want[i].abs(),
                "n {n}, batch {batch}, element {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
fn blend_scores_match_the_cpu_and_give_the_same_blend() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let small = Small::new(&dev, &module).unwrap();
    let mut rng = Lcg(3);
    for (count, width) in [(1, 5), (3, 32), (9, 2304)] {
        // Normalized candidates, as a blend is given them: sizes of about 1.
        let (query, candidates) = (rng.vec(width, 0.5), rng.vec(count * width, 1.7));
        let (dq, dc, ds) = (upload(&dev, &query), upload(&dev, &candidates), dev.alloc(count * 4).unwrap());
        small
            .blend_scores(
                Floats::first(&dq, width),
                Floats::first(&dc, count * width),
                Floats::first(&ds, count),
                count,
                width,
            )
            .unwrap();
        let got = download(&dev, &ds, count);
        let scale = 1.0 / (width as f32).sqrt();
        for j in 0..count {
            let c = &candidates[j * width..(j + 1) * width];
            let want = query.iter().zip(c).map(|(a, b)| a * b).sum::<f32>() * scale;
            let size: f32 = query.iter().zip(c).map(|(a, b)| (a * b).abs()).sum::<f32>() * scale;
            // `width` products added in a different order: the random-walk bar of the
            // matrix kernels' tests.
            let bound = 16.0 * ((width + 1) as f32).sqrt() * (f32::EPSILON / 2.0) * size;
            assert!((got[j] - want).abs() <= bound, "{count} of {width}, candidate {j}: {} vs {want}", got[j]);
        }
        // The CPU's whole blend for comparison of meaning: scores, softmax, weighted sum.
        let mut blended = vec![0.0f32; width];
        depthmix::mix_normalized(&query, &candidates, width, &mut blended);
        let mut weights = got.clone();
        cpu::softmax_inplace(&mut weights);
        for (i, want) in blended.iter().enumerate() {
            let mine: f32 = (0..count).map(|j| weights[j] * candidates[j * width + i]).sum();
            assert!((mine - want).abs() <= 1e-4, "blend element {i}: {mine} vs {want}");
        }
    }
}

#[test]
fn a_whole_blend_in_one_launch_matches_the_cpu() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let small = Small::new(&dev, &module).unwrap();
    let mut rng = Lcg(6);
    // One candidate (the blend is that candidate), a few, the most a launch takes; widths
    // below and above the number of threads.
    for (count, width, batch) in
        [(1, 5, 1), (2, 100, 3), (6, 2304, 1), (9, 2304, 16), (dluth_cuda::small::BLEND_MAX, 700, 2)]
    {
        // Each position has its own candidates; the query is shared.
        let (query, candidates) = (rng.vec(width, 0.6), rng.vec(batch * count * width, 1.7));
        let (dq, dc, dout) = (upload(&dev, &query), upload(&dev, &candidates), dev.alloc(batch * width * 4).unwrap());
        small
            .blend_mix(
                Floats::first(&dq, width),
                Floats::first(&dc, batch * count * width),
                Floats::first(&dout, batch * width),
                count,
                width,
            )
            .unwrap();
        let got = download(&dev, &dout, batch * width);
        // The scores are sums of `width` products added in another order, which moves the
        // weights by a few roundings times the size of a score; the result is a sum of
        // `count` terms of size below 2.
        let bound = 64.0 * (width as f32).sqrt() * f32::EPSILON;
        for (b, mine) in candidates.chunks_exact(count * width).enumerate() {
            let mut want = vec![0.0f32; width];
            depthmix::mix_normalized(&query, mine, width, &mut want);
            for i in 0..width {
                let g = got[b * width + i];
                assert!(
                    (g - want[i]).abs() <= bound,
                    "{count} of {width}, position {b}, element {i}: {g} vs {}",
                    want[i]
                );
            }
        }
    }
}

#[test]
fn several_vectors_normalized_in_one_launch_match_the_cpu() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let small = Small::new(&dev, &module).unwrap();
    let mut rng = Lcg(7);
    for (count, width, batch) in [(1, 3, 1), (2, 2304, 1), (2, 2304, 16), (dluth_cuda::small::MANY, 513, 3)] {
        // Each candidate in its own buffer, a batch of vectors of very different sizes; some
        // all zeros.
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|j| (0..batch).flat_map(|b| rng.vec(width, [0.0, 0.01, 3.0, 400.0][(j + b) % 4])).collect())
            .collect();
        let buffers: Vec<Buffer> = vectors.iter().map(|v| upload(&dev, v)).collect();
        let runs: Vec<Floats> = buffers.iter().map(|b| Floats::first(b, batch * width)).collect();
        let dout = dev.alloc(batch * count * width * 4).unwrap();
        small.normalize_many(&runs, Floats::first(&dout, batch * count * width), 1e-6, batch).unwrap();
        let got = download(&dev, &dout, batch * count * width);
        for (j, v) in vectors.iter().enumerate() {
            for b in 0..batch {
                let mut want = vec![0.0f32; width];
                cpu::rmsnorm_plain(&v[b * width..(b + 1) * width], 1e-6, &mut want);
                for i in 0..width {
                    // Position b's candidates are one after another.
                    let g = got[(b * count + j) * width + i];
                    assert!(
                        (g - want[i]).abs() <= 16.0 * f32::EPSILON * want[i].abs(),
                        "{count} of {width}, candidate {j} position {b} element {i}: {g} vs {}",
                        want[i]
                    );
                }
            }
        }
    }
}

#[test]
fn a_copy_is_exact() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let small = Small::new(&dev, &module).unwrap();
    let mut rng = Lcg(8);
    let x = rng.vec(3000, 9.0);
    let (dx, dout) = (upload(&dev, &x), dev.alloc(1000 * 4).unwrap());
    small.copy(Floats { buffer: &dx, at: 700, len: 1000 }, Floats::first(&dout, 1000)).unwrap();
    assert_eq!(download(&dev, &dout, 1000), x[700..1700]);
}

#[test]
fn the_merging_gate_and_the_soft_cap_match_the_cpu() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let small = Small::new(&dev, &module).unwrap();
    let mut rng = Lcg(4);
    let n = 2307;
    let (x, m, f, g) = (rng.vec(n, 3.0), rng.vec(n, 3.0), rng.vec(n, 4.0), rng.vec(n, 4.0));
    let (dx, dm, df, dg, dout) =
        (upload(&dev, &x), upload(&dev, &m), upload(&dev, &f), upload(&dev, &g), dev.alloc(n * 4).unwrap());
    let run = |b| Floats::first(b, n);
    small.gate_merge(run(&dx), run(&dm), run(&df), run(&dg), run(&dout)).unwrap();
    let got = download(&dev, &dout, n);
    for i in 0..n {
        let want = (1.0 - f[i].tanh()) * x[i] + g[i].tanh() * m[i];
        // Two products of a tanh, each possibly rounded differently in its last bits.
        let size = ((1.0 - f[i].tanh()) * x[i]).abs() + (g[i].tanh() * m[i]).abs();
        assert!(
            (got[i] - want).abs() <= 8.0 * f32::EPSILON * size.max(f32::MIN_POSITIVE),
            "gate element {i}: {} vs {want}",
            got[i]
        );
    }

    let scores = rng.vec(n, 90.0);
    let ds = upload(&dev, &scores);
    small.softcap(run(&ds), 30.0).unwrap();
    let got = download(&dev, &ds, n);
    for i in 0..n {
        let want = cpu::softcap(scores[i], 30.0);
        assert!(
            (got[i] - want).abs() <= 8.0 * f32::EPSILON * want.abs().max(f32::MIN_POSITIVE),
            "cap element {i}: {} vs {want}",
            got[i]
        );
    }
    assert!(got.iter().all(|v| v.abs() <= 30.0) && got.iter().any(|v| v.abs() > 29.0));
}

#[test]
fn requests_that_would_leave_the_buffers_are_refused() {
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let small = Small::new(&dev, &module).unwrap();
    let (a, b, c) = (dev.alloc(16 * 4).unwrap(), dev.alloc(16 * 4).unwrap(), dev.alloc(64 * 4).unwrap());
    let f = Floats::first;
    let refused = |what: &str, r: Result<(), CudaError>| {
        assert!(matches!(r, Err(CudaError::Refused(_))), "{what} was not refused")
    };
    refused("an input shorter than the output", small.add(f(&a, 16), f(&b, 12), f(&c, 16)));
    refused("more values than a buffer holds", small.add(f(&a, 17), f(&b, 17), f(&c, 17)));
    refused("adding in place", small.add(f(&a, 16), f(&b, 16), f(&a, 16)));
    refused("nothing to add", small.add(f(&a, 0), f(&b, 0), f(&c, 0)));
    refused("a negative eps", small.rmsnorm(f(&a, 16), f(&b, 16), f(&c, 16), -1.0));
    refused("normalizing in place", small.rmsnorm(f(&a, 16), f(&b, 16), f(&a, 16), 1e-6));
    refused("more candidates than the buffer holds", small.blend_scores(f(&a, 16), f(&c, 80), f(&b, 5), 5, 16));
    refused(
        "scores inside the candidates",
        small.blend_scores(f(&a, 16), f(&c, 64), Floats { buffer: &c, at: 4, len: 4 }, 4, 16),
    );
    refused(
        "a gate written over its input",
        small.gate_merge(f(&a, 16), f(&b, 16), f(&c, 16), Floats { buffer: &c, at: 16, len: 16 }, f(&b, 16)),
    );
    refused("a blend of more candidates than a launch takes", small.blend_mix(f(&a, 2), f(&c, 34), f(&b, 2), 17, 2));
    refused("a blend written over its candidates", small.blend_mix(f(&a, 16), f(&c, 64), f(&c, 16), 4, 16));
    refused("a blend of candidates the buffer does not hold", small.blend_mix(f(&a, 16), f(&c, 80), f(&b, 16), 5, 16));
    let nine = [f(&a, 4); 9];
    refused("more vectors than a launch takes", small.normalize_many(&nine, f(&c, 36), 1e-6, 1));
    refused("no vectors", small.normalize_many(&[], f(&c, 16), 1e-6, 1));
    refused("an output the vectors do not divide", small.normalize_many(&[f(&a, 5), f(&b, 5)], f(&c, 11), 1e-6, 1));
    refused("a vector normalized over itself", small.normalize_many(&[f(&c, 16)], f(&c, 16), 1e-6, 1));
    refused("a batch the vectors do not hold", small.normalize_many(&[f(&a, 16), f(&b, 16)], f(&c, 64), 1e-6, 2));
    refused("no positions", small.normalize_many(&[f(&a, 16)], f(&c, 16), 1e-6, 0));
    refused("weights that do not divide the vectors", small.rmsnorm(f(&a, 16), f(&b, 5), f(&c, 16), 1e-6));
    refused("a copy over itself", small.copy(f(&a, 16), f(&a, 16)));
    refused("a cap of zero", small.softcap(f(&a, 16), 0.0));
    refused("a cap that is not a number", small.softcap(f(&a, 16), f32::NAN));
}
