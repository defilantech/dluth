//! The Inkling forward pass with its matrix products on the GPU, against the same recorded
//! runs of the reference implementation that the CPU model is checked with.
//! GPU tests; they need an NVIDIA GPU of the architecture the kernels were built for.

#[path = "../../dluth-model/tests/common/mod.rs"]
mod common;
#[path = "../../dluth-model/tests/tiny_head/mod.rs"]
mod tiny_head;

use dluth_cuda::inkling::CudaProducts;
use dluth_cuda::{CudaDevice, Module};
use dluth_model::inkling::{load_config, InklingCpu};
use dluth_model::weights::Weights;
use std::path::{Path, PathBuf};

#[test]
fn the_tiny_model_matches_the_reference_with_products_on_the_gpu() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    // The same bar as the CPU model's test: four times the reference's own difference.
    let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
    assert!(tol > 0.0 && tol < 0.005, "tolerance {tol}");
    for case in golden["cases"].as_array().unwrap() {
        let products = CudaProducts::new(&dev, &module).unwrap();
        let mut model =
            InklingCpu::with_products(load_config(&dir).unwrap(), &weights, 64, Box::new(products)).unwrap();
        let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
        eprintln!("{}: {report:?}", case["prompt"]);
        assert_eq!(report.tie_at, None, "{}", case["prompt"]);
        assert_eq!(report.tokens_checked, 24, "{}", case["prompt"]);
    }
}

fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = x.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    x.iter().map(|v| v - max - log_sum).collect()
}

/// Text the recordings do not cover: 40 tokens of a fixed pattern, past the window, the
/// reach and the position where queries start to be scaled. The CPU model, which is checked
/// against the reference, stands in for the reference here, with the same bar: the five most
/// likely tokens' log-probabilities within four times the reference's own difference from
/// itself, and the same most likely token except at a near-tie.
#[test]
fn the_tiny_model_on_the_gpu_stays_close_to_the_cpu_model() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
    let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let products = CudaProducts::new(&dev, &module).unwrap();
    let mut gpu = InklingCpu::with_products(load_config(&dir).unwrap(), &weights, 64, Box::new(products)).unwrap();
    let mut cpu = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
    let (mut worst, mut identical) = (0.0f32, true);
    for pos in 0..40 {
        let token = ((pos * 37 + 11) % 256) as u32;
        let (a, b) = (gpu.forward_token(token, pos).unwrap(), cpu.forward_token(token, pos).unwrap());
        identical &= a == b;
        let (a, b) = (log_softmax(&a), log_softmax(&b));
        let mut order: Vec<usize> = (0..b.len()).collect();
        order.sort_by(|&i, &j| b[j].total_cmp(&b[i]));
        for &i in &order[..5] {
            worst = worst.max((a[i] - b[i]).abs());
            assert!((a[i] - b[i]).abs() < tol, "position {pos} token {i}: {} vs {}", a[i], b[i]);
        }
        let top = (0..a.len()).max_by(|&i, &j| a[i].total_cmp(&a[j])).unwrap();
        assert!(top == order[0] || b[order[0]] - b[order[1]] < tol, "position {pos}: {top} vs {}", order[0]);
    }
    eprintln!("largest log-probability difference between the two: {worst} (bar {tol})");
    assert!(!identical, "bit-identical scores would mean the GPU was not used");
}

#[test]
fn the_real_model_matches_the_reference_with_products_on_the_gpu() {
    let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
        eprintln!("skip: DLUTH_INKLING2_MINI not set");
        return;
    };
    let dir = PathBuf::from(dir);
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
    assert_eq!(golden["reference_dtype"], "float32");
    let noise = golden["reference_self_check"]["top5"].as_f64().unwrap() as f32;
    let tol = 4.0 * noise;
    assert!(noise > 0.0 && noise < 5e-5, "tolerance {tol} from a self-difference of {noise}");
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    for case in golden["cases"].as_array().unwrap() {
        let products = CudaProducts::new(&dev, &module).unwrap();
        let mut model =
            InklingCpu::with_products(load_config(&dir).unwrap(), &weights, 256, Box::new(products)).unwrap();
        let started = std::time::Instant::now();
        let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
        eprintln!("golden {}: {report:?} in {:.1} s", case["prompt"], started.elapsed().as_secs_f64());
        assert_eq!(report.tie_at, None, "{}", case["prompt"]);
        assert_eq!(report.tokens_checked, 32, "{}", case["prompt"]);
    }
}

/// The same three checks with every value on the device (`CudaOps`), not only the products.
mod on_the_device {
    use super::*;
    use dluth_cuda::ops::CudaOps;
    use dluth_model::inkling::Inkling;

    #[test]
    fn the_tiny_model_matches_the_reference() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
        assert!(tol > 0.0 && tol < 0.005, "tolerance {tol}");
        for case in golden["cases"].as_array().unwrap() {
            let ops = CudaOps::new(&dev, &module).unwrap();
            let mut model = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
            let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
            eprintln!("device, {}: {report:?}", case["prompt"]);
            assert_eq!(report.tie_at, None, "{}", case["prompt"]);
            assert_eq!(report.tokens_checked, 24, "{}", case["prompt"]);
        }
    }

    /// Unrecorded text, twice over, so that a second sequence on the same model (position 0
    /// again) is covered: nothing of the first may leak into it.
    #[test]
    fn the_tiny_model_stays_close_to_the_cpu_model_across_two_sequences() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
        let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let ops = CudaOps::new(&dev, &module).unwrap();
        let mut gpu = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
        let mut cpu = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
        let mut worst = 0.0f32;
        for (sequence, length) in [(0usize, 40usize), (1, 25)] {
            for pos in 0..length {
                let token = ((pos * 37 + 11 + sequence * 101) % 256) as u32;
                let (a, b) = (gpu.forward_token(token, pos).unwrap(), cpu.forward_token(token, pos).unwrap());
                let (a, b) = (log_softmax(&a), log_softmax(&b));
                let mut order: Vec<usize> = (0..b.len()).collect();
                order.sort_by(|&i, &j| b[j].total_cmp(&b[i]));
                for &i in &order[..5] {
                    worst = worst.max((a[i] - b[i]).abs());
                    assert!(
                        (a[i] - b[i]).abs() < tol,
                        "sequence {sequence} position {pos} token {i}: {} vs {}",
                        a[i],
                        b[i]
                    );
                }
                let top = (0..a.len()).max_by(|&i, &j| a[i].total_cmp(&a[j])).unwrap();
                assert!(top == order[0] || b[order[0]] - b[order[1]] < tol, "sequence {sequence} position {pos}");
            }
        }
        eprintln!("device: largest log-probability difference from the CPU model: {worst} (bar {tol})");
    }

    /// Every device buffer the model and its operations took is given back when they go:
    /// weights, parameters, pooled vectors, and each layer pass's histories and rings.
    #[test]
    fn a_dropped_model_gives_every_device_buffer_back() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let before = dev.live_buffers();
        {
            let ops = CudaOps::new(&dev, &module).unwrap();
            let mut model = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
            for pos in 0..20 {
                model.forward_token((pos * 7 % 250) as u32, pos).unwrap();
            }
            assert!(dev.live_buffers() > before + 100, "the model should be holding buffers while it lives");
        }
        assert_eq!(dev.live_buffers(), before);
    }

    /// The five most likely tokens' log-probabilities within `tol`, and the same most likely
    /// token except at a near-tie: the bar the model is held to against the reference.
    /// Returns the largest difference among those five.
    fn close_scores(a: &[f32], b: &[f32], tol: f32, what: &str) -> f32 {
        let (a, b) = (log_softmax(a), log_softmax(b));
        let mut order: Vec<usize> = (0..b.len()).collect();
        order.sort_by(|&i, &j| b[j].total_cmp(&b[i]));
        let mut worst = 0.0f32;
        for &i in &order[..5] {
            worst = worst.max((a[i] - b[i]).abs());
            assert!((a[i] - b[i]).abs() < tol, "{what}, token {i}: {} vs {}", a[i], b[i]);
        }
        let top = (0..a.len()).max_by(|&i, &j| a[i].total_cmp(&a[j])).unwrap();
        assert!(top == order[0] || b[order[0]] - b[order[1]] < tol, "{what}: {top} vs {}", order[0]);
        worst
    }

    /// Two sequences fed in batches of every kind, each position's scores against the CPU
    /// model one token at a time. The tiny model's window is 8, so batches end on it,
    /// straddle it and are longer than it, and the histories of its local layers (8 + 15
    /// rows) wrap around inside batches. The second sequence starts again at position 0.
    #[test]
    fn the_tiny_model_in_batches_stays_close_to_the_cpu_model() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
        let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let ops = CudaOps::new(&dev, &module).unwrap();
        let stats = ops.stats();
        let cfg = load_config(&dir).unwrap();
        let (vocab, passes) = (cfg.vocab_out, cfg.layer_passes());
        let mut gpu = Inkling::with_ops(cfg, &weights, 64, ops).unwrap();
        let mut cpu = InklingCpu::new(load_config(&dir).unwrap(), &weights, 64).unwrap();
        let mut worst = 0.0f32;
        let splits: [&[usize]; 2] = [&[16, 1, 7, 2, 16, 5, 13, 4], &[3, 8, 1, 1, 16, 11]];
        for (sequence, split) in splits.into_iter().enumerate() {
            let mut pos = 0;
            for &n in split {
                let tokens: Vec<u32> = (pos..pos + n).map(|p| ((p * 37 + 11 + sequence * 101) % 256) as u32).collect();
                let waits = stats.borrow().waits;
                let got = gpu.forward_tokens(&tokens, pos).unwrap();
                // However many positions: one wait for the embeddings, one per layer pass
                // for the router's scores, one for the scores of the vocabulary.
                assert_eq!(stats.borrow().waits - waits, passes + 2, "a batch of {n}");
                assert!(stats.borrow().waiting > std::time::Duration::ZERO, "the waits are timed");
                assert_eq!(got.len(), n * vocab);
                for (i, scores) in got.chunks_exact(vocab).enumerate() {
                    let want = cpu.forward_token(tokens[i], pos + i).unwrap();
                    let what = format!("sequence {sequence}, position {} in a batch of {n}", pos + i);
                    worst = worst.max(close_scores(scores, &want, tol, &what));
                }
                pos += n;
            }
        }
        eprintln!("device in batches: largest log-probability difference from the CPU model: {worst} (bar {tol})");
    }

    /// Feeding without scoring, then asking: the scores after the last token only, the same
    /// as one token at a time on the device gives.
    #[test]
    fn the_tiny_model_scores_the_last_position_of_a_fed_prompt() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
        let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let model =
            || Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, CudaOps::new(&dev, &module).unwrap());
        let (mut batched, mut single) = (model().unwrap(), model().unwrap());
        let tokens: Vec<u32> = (0..37).map(|p| ((p * 53 + 5) % 256) as u32).collect();
        let mut want = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            want = single.forward_token(t, pos).unwrap();
        }
        for (i, chunk) in tokens.chunks(16).enumerate() {
            batched.feed(chunk, i * 16).unwrap();
        }
        close_scores(&batched.next_scores().unwrap(), &want, tol, "after a prompt of 37 fed in batches of 16");
        // Generation with the prompt fed in batches picks the same tokens.
        let alone = single.generate_greedy(&tokens[..21], 10, u32::MAX).unwrap();
        batched.set_prompt_batch(16).unwrap();
        assert_eq!(batched.generate_greedy(&tokens[..21], 10, u32::MAX).unwrap(), alone);
    }

    /// Sweeps of a few true tokens followed by wrong ones, which are then undone: every true
    /// position's scores against the CPU model fed the true tokens alone, over more tokens
    /// than the local layers' histories have rows.
    #[test]
    fn undone_tokens_leave_no_trace_on_the_device() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
        let tol = 4.0 * golden["reference_self_difference"].as_f64().unwrap() as f32;
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let cfg = load_config(&dir).unwrap();
        let vocab = cfg.vocab_out;
        let mut gpu = Inkling::with_ops(cfg, &weights, 128, CudaOps::new(&dev, &module).unwrap()).unwrap();
        let mut cpu = InklingCpu::new(load_config(&dir).unwrap(), &weights, 128).unwrap();
        let text: Vec<u32> = (0..100).map(|p| ((p * 37 + 11) % 256) as u32).collect();
        let sweeps = [(1usize, 3usize), (1, 15), (4, 1), (2, 0), (1, 1), (3, 7), (1, 0), (5, 11)];
        let (mut pos, mut turn, mut worst) = (0, 0, 0.0f32);
        while pos < text.len() {
            let (right, wrong) = sweeps[turn % sweeps.len()];
            turn += 1;
            let right = right.min(text.len() - pos);
            let mut batch = text[pos..pos + right].to_vec();
            batch.extend((0..wrong).map(|i| ((pos * 13 + i * 5 + 200) % 256) as u32));
            let got = gpu.forward_tokens(&batch, pos).unwrap();
            let mut want = Vec::new();
            for i in 0..right {
                want = cpu.forward_token(text[pos + i], pos + i).unwrap();
                let what = format!("position {} in sweep {turn}", pos + i);
                worst = worst.max(close_scores(&got[i * vocab..(i + 1) * vocab], &want, tol, &what));
            }
            gpu.undo(wrong).unwrap();
            pos += right;
            close_scores(&gpu.next_scores().unwrap(), &want, tol, &format!("after undoing in sweep {turn}"));
        }
        eprintln!("device with undone tokens: largest log-probability difference from the CPU model: {worst}");
        // A whole batch of 16 cannot be undone down to nothing, nor 16 tokens at once.
        gpu.feed(&text[..16], 0).unwrap();
        assert!(gpu.undo(16).is_err());
        gpu.undo(15).unwrap();
        assert_eq!(gpu.next_position(), 1);
        // A small batch keeps its first token too. Nothing on the device would notice if it
        // did not: the model refuses.
        gpu.feed(&text[1..5], 1).unwrap();
        assert!(gpu.undo(4).is_err());
        gpu.undo(3).unwrap();
        assert_eq!(gpu.next_position(), 2);
    }

    /// Generation with guesses on the device gives the tokens of generation without, with
    /// right guesses, wrong guesses and guesses from the text itself.
    #[test]
    fn generation_with_guesses_gives_the_same_tokens_on_the_device() {
        use dluth_model::inkling::{Drafter, LookupDrafter};
        struct Scripted(Vec<u32>, usize, usize);
        impl<V> Drafter<V> for Scripted {
            fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
                let done = tokens.len() - self.1;
                let wrong_every = self.2;
                self.0[done.min(self.0.len())..]
                    .iter()
                    .take(most)
                    .enumerate()
                    .map(
                        |(i, &t)| {
                            if wrong_every != 0 && (done + i).is_multiple_of(wrong_every) {
                                (t + 1) % 256
                            } else {
                                t
                            }
                        },
                    )
                    .collect()
            }
        }
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let mut gpu =
            Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, CudaOps::new(&dev, &module).unwrap()).unwrap();
        let prompt: Vec<u32> = (41..64).map(|p| ((p * 37 + 11) % 256) as u32).collect();
        let want = gpu.generate_greedy(&prompt, 40, u32::MAX).unwrap();
        let (got, stats) = gpu
            .generate_greedy_drafted(&prompt, 40, u32::MAX, &mut Scripted(want.clone(), prompt.len(), 0), 15, |_| {})
            .unwrap();
        assert_eq!(got, want);
        assert_eq!((stats.sweeps, stats.accepted), (3, 36));
        let (got, stats) = gpu
            .generate_greedy_drafted(&prompt, 40, u32::MAX, &mut Scripted(want.clone(), prompt.len(), 3), 5, |_| {})
            .unwrap();
        assert_eq!(got, want);
        assert!(stats.accepted > 5 && stats.accepted < stats.drafted, "{stats:?}");
        let (got, _) =
            gpu.generate_greedy_drafted(&prompt, 40, u32::MAX, &mut LookupDrafter::new(3, 1), 4, |_| {}).unwrap();
        assert_eq!(got, want);
    }

    /// A blend of more candidates than one launch takes goes step by step. No model has
    /// that many, so the path is driven directly: 17 and 5 candidates, each a batch of
    /// three positions, blended for two queries (one with a normalization after it), read
    /// back through the output table, against the CPU operations on the same values.
    #[test]
    fn a_blend_of_more_candidates_than_a_launch_takes_matches_the_cpu() {
        use dluth_core::bf16_to_f32;
        use dluth_model::inkling::{CpuOps, CpuProducts, Ops};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let weights = Weights::open_dir(&dir).unwrap();
        let cfg = load_config(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let mut gpu = CudaOps::new(&dev, &module).unwrap();
        let mut cpu = CpuOps::new(Box::new(CpuProducts));
        let table = weights.bytes("model.llm.embed.weight").unwrap();
        let unembed = weights.bytes("model.llm.unembed.weight").unwrap();
        let floats = |name: &str| -> Vec<f32> {
            weights
                .bytes(name)
                .unwrap()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
                .collect()
        };
        let embed_norm = floats("model.llm.embed_norm.weight");
        let after = floats("model.llm.norm.weight");
        let h = cfg.hidden;
        let queries: [Vec<f32>; 2] = [
            (0..h).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.3).collect(),
            (0..h).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.4).collect(),
        ];
        for count in [17usize, 5] {
            // Each candidate is three positions' embeddings of other tokens.
            let tokens =
                |j: usize| -> [u32; 3] { [(j * 3 % 250) as u32, (j * 7 % 250 + 1) as u32, (j * 11 % 250 + 2) as u32] };
            let on_gpu: Vec<_> = (0..count).map(|j| gpu.embed(&cfg, table, &embed_norm, &tokens(j)).unwrap()).collect();
            let on_cpu: Vec<_> = (0..count).map(|j| cpu.embed(&cfg, table, &embed_norm, &tokens(j)).unwrap()).collect();
            let gpu_blend = gpu.normalize(&on_gpu.iter().collect::<Vec<_>>()).unwrap();
            let cpu_blend = cpu.normalize(&on_cpu.iter().collect::<Vec<_>>()).unwrap();
            let asked: [&[f32]; 2] = [&queries[0], &queries[1]];
            let norms = [Some(&after[..]), None];
            let got = gpu.mix_many(&gpu_blend, &asked, &norms, cfg.eps).unwrap();
            let want = cpu.mix_many(&cpu_blend, &asked, &norms, cfg.eps).unwrap();
            for q in 0..2 {
                let (g, w) =
                    (gpu.scores(&cfg, unembed, &got[q]).unwrap(), cpu.scores(&cfg, unembed, &want[q]).unwrap());
                assert_eq!(g.len(), 3 * cfg.vocab_out);
                // Scores are sums of 32 products of values of size about 1.
                let worst = g.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(worst < 2e-3, "{count} candidates, query {q}: scores differ by {worst}");
                // The three positions give different scores.
                assert!(w[..cfg.vocab_out] != w[cfg.vocab_out..2 * cfg.vocab_out]);
            }
            // And so do the two queries.
            assert!(cpu.scores(&cfg, unembed, &want[0]).unwrap() != cpu.scores(&cfg, unembed, &want[1]).unwrap());
        }
    }

    /// A drafter that passes everything on and keeps what was proposed.
    struct Recording<D>(D, Vec<Vec<u32>>);
    impl<V, D: dluth_model::inkling::Drafter<V>> dluth_model::inkling::Drafter<V> for Recording<D> {
        fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
            let guesses = self.0.draft(tokens, most);
            self.1.push(guesses.clone());
            guesses
        }
        fn wants(&self) -> Vec<usize> {
            self.0.wants()
        }
        fn fed(&mut self, first: usize, kept: usize, states: &[V]) -> Result<(), dluth_model::ModelError> {
            self.0.fed(first, kept, states)
        }
    }

    fn embed_norm_of(weights: &Weights) -> Vec<f32> {
        let bytes = weights.bytes("model.llm.embed_norm.weight").unwrap();
        bytes.as_chunks::<2>().0.iter().map(|c| dluth_core::bf16_to_f32(u16::from_le_bytes(*c))).collect()
    }

    /// The draft head on the device against the head on the CPU, both of random weights on
    /// the tiny model: the same text as plain generation, and the same proposals sweep
    /// after sweep. The proposals come from the scores of the whole vocabulary, so two that
    /// agree had every step of the head agree closely.
    #[test]
    fn the_draft_head_on_the_device_proposes_what_the_cpu_head_proposes() {
        device_head_against_cpu_head("device", &[], 0);
        // As in the real head: a local layer whose window is far shorter than the text, and
        // a layer that sees everything. Then every layer local, with a window shorter than
        // the block, so that the last block positions see none of the text.
        device_head_against_cpu_head("device-local", &[0], 6);
        device_head_against_cpu_head("device-all-local", &[0, 1], 3);
    }

    fn device_head_against_cpu_head(name: &str, local: &[usize], window: usize) {
        use dluth_cuda::draft_head::CudaDraftHead;
        use dluth_model::inkling::mtp::{MtpDrafter, MtpHead};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let weights = Weights::open_dir(&dir).unwrap();
        let cfg = load_config(&dir).unwrap();
        let scratch = tiny_head::Scratch::new(name);
        let head_cfg = tiny_head::tiny_local_config(cfg.hidden, cfg.layer_passes(), local, window);
        tiny_head::write_head(&scratch.0, &head_cfg, cfg.vocab, None, None);
        let embed = weights.bytes("model.llm.embed.weight").unwrap();
        let unembed = &weights.bytes("model.llm.unembed.weight").unwrap()[..cfg.vocab_out * cfg.hidden * 2];
        let prompt: Vec<u32> = (41..64).map(|p| ((p * 37 + 11) % 256) as u32).collect();
        // The text is longer than the window from the first guess on: a local layer is given
        // the text from a position that is not its start.
        assert!(local.is_empty() || prompt.len() > window + head_cfg.block);

        let cpu_head = MtpHead::load(&scratch.0, head_cfg.clone()).unwrap();
        let mut on_cpu = Recording(
            MtpDrafter::new(&cpu_head, embed, embed_norm_of(&weights), unembed, cfg.vocab_out).unwrap(),
            Vec::new(),
        );
        let mut cpu = InklingCpu::new(load_config(&dir).unwrap(), &weights, 96).unwrap();
        let want = cpu.generate_greedy(&prompt, 40, u32::MAX).unwrap();
        let (cpu_text, cpu_stats) = cpu.generate_greedy_drafted(&prompt, 40, u32::MAX, &mut on_cpu, 3, |_| {}).unwrap();
        assert_eq!(cpu_text, want);

        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let mut ops = CudaOps::new(&dev, &module).unwrap();
        let table = ops.device_weight(unembed).unwrap();
        let before = dev.live_buffers();
        {
            let head = CudaDraftHead::load(
                &dev,
                &module,
                &scratch.0,
                head_cfg.clone(),
                embed,
                embed_norm_of(&weights),
                table,
                cfg.vocab_out,
                96,
            )
            .unwrap();
            let mut on_gpu = Recording(head, Vec::new());
            let mut gpu = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 96, ops).unwrap();
            let plain = gpu.generate_greedy(&prompt, 40, u32::MAX).unwrap();
            let (gpu_text, gpu_stats) =
                gpu.generate_greedy_drafted(&prompt, 40, u32::MAX, &mut on_gpu, 3, |_| {}).unwrap();
            assert_eq!(gpu_text, plain);
            assert_eq!(on_gpu.0.last_error, None);
            assert!(gpu_stats.drafted >= gpu_stats.sweeps * 2, "the head proposes at every sweep: {gpu_stats:?}");
            // Where the two models chose the same text, the two heads were asked the same
            // questions, and must have answered alike nearly always (a near-tie between two
            // tokens' scores can fall either way).
            if plain == want {
                assert_eq!(on_gpu.1.len(), on_cpu.1.len());
                let same = on_gpu.1.iter().zip(&on_cpu.1).filter(|(a, b)| a == b).count();
                eprintln!("{name} head: {same} of {} proposals identical to the CPU head's; {gpu_stats:?} against {cpu_stats:?}", on_cpu.1.len());
                assert!(same * 10 >= on_cpu.1.len() * 9, "{same} of {}", on_cpu.1.len());
            } else {
                eprintln!("the two models' texts differ at a near-tie; the heads were not compared");
            }
            // A second text starts the head afresh; a text it was not told of gets no proposal.
            let (again, _) = gpu.generate_greedy_drafted(&prompt, 12, u32::MAX, &mut on_gpu, 3, |_| {}).unwrap();
            assert_eq!(again, plain[..12]);
            use dluth_model::inkling::Drafter;
            assert!(Drafter::<dluth_cuda::ops::DeviceVec>::draft(&mut on_gpu.0, &prompt[..5], 3).is_empty());
            assert!(Drafter::<dluth_cuda::ops::DeviceVec>::fed(&mut on_gpu.0, 3, 1, &[]).is_err());
        }
        // Head and model are gone, and with them more buffers than the head ever took: a head
        // that kept its buffers would leave the count above where it was.
        assert!(dev.live_buffers() < before);
    }

    /// The real head on the real model on the device: the same text as plain generation,
    /// in well under half the sweeps.
    #[test]
    fn the_real_draft_head_guesses_well_on_the_device() {
        use dluth_cuda::draft_head::CudaDraftHead;
        use dluth_model::inkling::mtp::load_mtp_config;
        let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
            eprintln!("skip: DLUTH_INKLING2_MINI not set");
            return;
        };
        let dir = PathBuf::from(dir);
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
        let prompt: Vec<u32> = serde_json::from_value(golden["cases"][1]["prompt_ids"].clone()).unwrap();
        let weights = Weights::open_dir(&dir).unwrap();
        let cfg = load_config(&dir).unwrap();
        let (eos, vocab_out, h) = (cfg.eos, cfg.vocab_out, cfg.hidden);
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let mut ops = CudaOps::new(&dev, &module).unwrap();
        let unembed = &weights.bytes("model.llm.unembed.weight").unwrap()[..vocab_out * h * 2];
        let table = ops.device_weight(unembed).unwrap();
        let mut head = CudaDraftHead::load(
            &dev,
            &module,
            &dir,
            load_mtp_config(&dir).unwrap().expect("a draft head"),
            weights.bytes("model.llm.embed.weight").unwrap(),
            embed_norm_of(&weights),
            table,
            vocab_out,
            256,
        )
        .unwrap();
        let mut model = Inkling::with_ops(cfg, &weights, 256, ops).unwrap();
        let new = 96;
        let want = model.generate_greedy(&prompt, new, eos).unwrap();
        let (got, stats) = model.generate_greedy_drafted(&prompt, new, eos, &mut head, 5, |_| {}).unwrap();
        eprintln!("the real head on the device: {} tokens, {stats:?}", got.len());
        assert_eq!(head.last_error, None);
        assert_eq!(got, want);
        assert!(stats.sweeps * 2 < want.len() - 1, "{stats:?} for {} tokens", want.len());
    }

    /// The real head after a text longer than the window of its local layers: its guesses
    /// must stay good. A head whose local layers look at the whole text guesses well after
    /// a short text and badly after a long one.
    #[test]
    fn the_real_draft_head_guesses_well_after_a_long_text() {
        use dluth_cuda::draft_head::CudaDraftHead;
        use dluth_model::inkling::mtp::load_mtp_config;
        use dluth_model::tokenizer::Tokenizer;
        let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
            eprintln!("skip: DLUTH_INKLING2_MINI not set");
            return;
        };
        let dir = PathBuf::from(dir);
        let head_cfg = load_mtp_config(&dir).unwrap().expect("a draft head");
        // A long text that is always at hand: this file.
        let text = include_str!("inkling_golden_cuda.rs");
        let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
        let mut prompt = tokenizer.encode(text).unwrap();
        let long = 3 * head_cfg.window;
        assert!(!head_cfg.local.is_empty() && prompt.len() >= long, "{} tokens, {head_cfg:?}", prompt.len());
        prompt.truncate(long);
        let weights = Weights::open_dir(&dir).unwrap();
        let cfg = load_config(&dir).unwrap();
        let (eos, vocab_out, h) = (cfg.eos, cfg.vocab_out, cfg.hidden);
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let mut ops = CudaOps::new(&dev, &module).unwrap();
        let unembed = &weights.bytes("model.llm.unembed.weight").unwrap()[..vocab_out * h * 2];
        let table = ops.device_weight(unembed).unwrap();
        let (new, room) = (128, long + 128 + 16);
        let mut head = CudaDraftHead::load(
            &dev,
            &module,
            &dir,
            head_cfg,
            weights.bytes("model.llm.embed.weight").unwrap(),
            embed_norm_of(&weights),
            table,
            vocab_out,
            room,
        )
        .unwrap();
        // As the server runs it: the prompt read in wide sweeps, of which the head is told
        // part by part, and the experts many positions chose multiplied on the tensor cores.
        let mut model = Inkling::with_ops(cfg, &weights, room, dluth_cuda::wide::WideOps::new(ops)).unwrap();
        model.set_prompt_batch(256).unwrap();
        let (got, stats) = model.generate_greedy_drafted(&prompt, new, eos, &mut head, 2, |_| {}).unwrap();
        eprintln!("the real head after {long} tokens: {} tokens, {stats:?}", got.len());
        assert_eq!(head.last_error, None);
        assert!(stats.accepted * 5 >= stats.drafted * 2, "{stats:?}");
    }

    /// The head beside a model that reads in wide sweeps is told of a sweep part by part. On
    /// the tiny model, where wide sweeps compute what batches compute to the last bit, it
    /// must then propose exactly what it proposes beside batches, sweep after sweep.
    #[test]
    fn the_draft_head_beside_wide_sweeps_proposes_what_it_proposes_beside_batches() {
        use dluth_cuda::draft_head::CudaDraftHead;
        use dluth_cuda::wide::WideOps;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let weights = Weights::open_dir(&dir).unwrap();
        let cfg = load_config(&dir).unwrap();
        let scratch = tiny_head::Scratch::new("device-wide");
        let head_cfg = tiny_head::tiny_local_config(cfg.hidden, cfg.layer_passes(), &[0], 6);
        tiny_head::write_head(&scratch.0, &head_cfg, cfg.vocab, None, None);
        let embed = weights.bytes("model.llm.embed.weight").unwrap();
        let unembed = &weights.bytes("model.llm.unembed.weight").unwrap()[..cfg.vocab_out * cfg.hidden * 2];
        // 41 tokens: a sweep of 16, 16 and 9 where batches are 16, 16 and 9 too, and sweeps
        // of 20 (16 and 4) where the parts fall elsewhere than the batches.
        let prompt: Vec<u32> = (0..41).map(|p| ((p * 37 + 11) % 256) as u32).collect();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let head_for = |ops: &mut CudaOps| {
            let table = ops.device_weight(unembed).unwrap();
            let embed_norm = embed_norm_of(&weights);
            CudaDraftHead::load(
                &dev,
                &module,
                &scratch.0,
                head_cfg.clone(),
                embed,
                embed_norm,
                table,
                cfg.vocab_out,
                96,
            )
            .unwrap()
        };
        let mut ops = CudaOps::new(&dev, &module).unwrap();
        let mut beside_batches = Recording(head_for(&mut ops), Vec::new());
        let mut plain = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 96, ops).unwrap();
        let (want, want_stats) =
            plain.generate_greedy_drafted(&prompt, 40, u32::MAX, &mut beside_batches, 3, |_| {}).unwrap();
        assert!(want_stats.drafted > 0 && beside_batches.1.iter().any(|guesses| !guesses.is_empty()));

        for sweep in [41, 20] {
            let mut ops = CudaOps::new(&dev, &module).unwrap();
            let mut beside_sweeps = Recording(head_for(&mut ops), Vec::new());
            let mut wide = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 96, WideOps::new(ops)).unwrap();
            wide.set_prompt_batch(sweep).unwrap();
            let (got, stats) =
                wide.generate_greedy_drafted(&prompt, 40, u32::MAX, &mut beside_sweeps, 3, |_| {}).unwrap();
            assert_eq!(beside_sweeps.0.last_error, None);
            assert_eq!(got, want, "sweeps of {sweep}");
            assert_eq!(beside_sweeps.1, beside_batches.1, "the proposals beside sweeps of {sweep}");
            assert_eq!((stats.drafted, stats.accepted), (want_stats.drafted, want_stats.accepted));
        }
    }

    /// The real model reading a text in wide sweeps, where most experts go to the tensor
    /// cores, against the same model fed one token at a time, where none does: the scores
    /// after the text at the bar of the recordings.
    #[test]
    fn the_real_model_scores_the_same_after_a_text_read_in_wide_sweeps() {
        use dluth_cuda::wide::WideOps;
        use dluth_model::tokenizer::Tokenizer;
        let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
            eprintln!("skip: DLUTH_INKLING2_MINI not set");
            return;
        };
        let dir = PathBuf::from(dir);
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
        let tol = 4.0 * golden["reference_self_check"]["top5"].as_f64().unwrap() as f32;
        // A text that is always at hand: this file. 300 tokens choose 2,400 experts of a
        // layer's 128: about 19 positions to an expert, and 4 in the sweeps of 64, where
        // the experts that go to the tensor cores are the ones neighbouring tokens share.
        let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
        let mut text = tokenizer.encode(include_str!("inkling_golden_cuda.rs")).unwrap();
        assert!(text.len() >= 300);
        text.truncate(300);
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let ops = WideOps::new(CudaOps::new(&dev, &module).unwrap());
        let mut model = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 320, ops).unwrap();
        let mut want = Vec::new();
        for (pos, &t) in text.iter().enumerate() {
            want = model.forward_token(t, pos).unwrap();
        }
        for sweep in [300, 128, 64] {
            for (i, chunk) in text.chunks(sweep).enumerate() {
                model.feed(chunk, i * sweep).unwrap();
            }
            let got = model.next_scores().unwrap();
            let worst = close_scores(&got, &want, tol, &format!("sweeps of {sweep}"));
            eprintln!("device, 300 tokens in sweeps of {sweep}: largest difference {worst} (bar {tol})");
            // Not the same numbers: the tensor cores were used.
            assert!(got != want, "sweeps of {sweep} gave one-at-a-time scores to the last bit");
        }
    }

    /// More positions than a launch takes are refused before anything is computed, and the
    /// model goes on as if it had not been asked.
    #[test]
    fn too_large_a_batch_is_refused_and_changes_nothing() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling");
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let ops = CudaOps::new(&dev, &module).unwrap();
        let mut gpu = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
        let ops = CudaOps::new(&dev, &module).unwrap();
        let mut fresh = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, ops).unwrap();
        assert_eq!(gpu.forward_token(7, 0).unwrap(), fresh.forward_token(7, 0).unwrap());
        let e = gpu.forward_tokens(&[9; 17], 1).unwrap_err();
        assert!(
            matches!(e, dluth_model::ModelError::Capacity(_)) && e.to_string().contains("1 to 16 positions"),
            "{e}"
        );
        assert_eq!(gpu.forward_token(8, 1).unwrap(), fresh.forward_token(8, 1).unwrap());
    }

    /// The real model with each recorded prompt fed in batches of 16: the scores after the
    /// prompt against the same model fed one token at a time, at the bar of the recordings.
    #[test]
    fn the_real_model_scores_the_same_after_a_prompt_fed_in_batches() {
        let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
            eprintln!("skip: DLUTH_INKLING2_MINI not set");
            return;
        };
        let dir = PathBuf::from(dir);
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
        let tol = 4.0 * golden["reference_self_check"]["top5"].as_f64().unwrap() as f32;
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        let ops = CudaOps::new(&dev, &module).unwrap();
        let mut model = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 256, ops).unwrap();
        for case in golden["cases"].as_array().unwrap() {
            let prompt: Vec<u32> = serde_json::from_value(case["prompt_ids"].clone()).unwrap();
            let mut want = Vec::new();
            let started = std::time::Instant::now();
            for (pos, &t) in prompt.iter().enumerate() {
                want = model.forward_token(t, pos).unwrap();
            }
            let one_at_a_time = started.elapsed();
            let started = std::time::Instant::now();
            for (i, chunk) in prompt.chunks(16).enumerate() {
                model.feed(chunk, i * 16).unwrap();
            }
            let got = model.next_scores().unwrap();
            let worst = close_scores(&got, &want, tol, &format!("{}", case["prompt"]));
            eprintln!(
                "device, prompt of {} tokens: {:.0} ms one at a time, {:.0} ms in batches of 16; largest difference {worst}",
                prompt.len(),
                one_at_a_time.as_secs_f64() * 1e3,
                started.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    #[test]
    fn the_real_model_matches_the_reference() {
        let Ok(dir) = std::env::var("DLUTH_INKLING2_MINI") else {
            eprintln!("skip: DLUTH_INKLING2_MINI not set");
            return;
        };
        let dir = PathBuf::from(dir);
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/inkling2-mini-greedy.json")).unwrap();
        assert_eq!(golden["reference_dtype"], "float32");
        let noise = golden["reference_self_check"]["top5"].as_f64().unwrap() as f32;
        let tol = 4.0 * noise;
        assert!(noise > 0.0 && noise < 5e-5, "tolerance {tol} from a self-difference of {noise}");
        let weights = Weights::open_dir(&dir).unwrap();
        let dev = CudaDevice::open(0).expect("GPU");
        let module = Module::load_embedded(&dev).unwrap();
        for case in golden["cases"].as_array().unwrap() {
            let ops = CudaOps::new(&dev, &module).unwrap();
            let mut model = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 256, ops).unwrap();
            let started = std::time::Instant::now();
            let report = common::check_case(&mut |token, pos| model.forward_token(token, pos).unwrap(), case, tol);
            eprintln!("device, golden {}: {report:?} in {:.1} s", case["prompt"], started.elapsed().as_secs_f64());
            assert_eq!(report.tie_at, None, "{}", case["prompt"]);
            assert_eq!(report.tokens_checked, 32, "{}", case["prompt"]);
        }
    }
}
