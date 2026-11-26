//! Sweeps of more positions than one launch takes (`WideOps`), on the device. A wide sweep
//! must compute exactly what the same positions compute sixteen at a time: only the expert
//! step works on the whole sweep, and it reads each expert once for every position that
//! chose it.

use dluth_cuda::ops::CudaOps;
use dluth_cuda::wide::{WideOps, MAX_SWEEP};
use dluth_cuda::{CudaDevice, Module};
use dluth_model::inkling::{load_config, Inkling};
use dluth_model::weights::Weights;
use dluth_model::ModelError;
use std::path::{Path, PathBuf};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-inkling")
}

fn tokens(count: usize) -> Vec<u32> {
    (0..count).map(|p| ((p * 37 + 11) % 256) as u32).collect()
}

#[test]
fn a_wide_sweep_scores_what_sixteen_positions_at_a_time_score() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let mut plain =
        Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, CudaOps::new(&dev, &module).unwrap()).unwrap();
    let wide_ops = WideOps::new(CudaOps::new(&dev, &module).unwrap());
    let mut wide = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, wide_ops).unwrap();
    let text = tokens(60);
    // 40 positions in one sweep (16, 16 and 8), then 3, then 17 (16 and 1).
    let mut pos = 0;
    for sweep in [40, 3, 17] {
        let got = wide.forward_tokens(&text[pos..pos + sweep], pos).unwrap();
        let (mut want, mut at) = (Vec::new(), pos);
        for part in text[pos..pos + sweep].chunks(16) {
            want.extend(plain.forward_tokens(part, at).unwrap());
            at += part.len();
        }
        assert_eq!(got.len(), want.len());
        let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "a sweep of {sweep} from position {pos}: {differing} scores differ");
        pos += sweep;
    }
}

#[test]
fn a_prompt_read_in_wide_sweeps_gives_the_same_text() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let mut plain =
        Inkling::with_ops(load_config(&dir).unwrap(), &weights, 96, CudaOps::new(&dev, &module).unwrap()).unwrap();
    let wide_ops = WideOps::new(CudaOps::new(&dev, &module).unwrap());
    let mut wide = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 96, wide_ops).unwrap();
    let prompt = tokens(45);
    let want = plain.generate_greedy(&prompt, 30, u32::MAX).unwrap();
    for sweep in [45, 40, 17, 1] {
        wide.set_prompt_batch(sweep).unwrap();
        assert_eq!(wide.generate_greedy(&prompt, 30, u32::MAX).unwrap(), want, "sweeps of {sweep}");
    }
}

#[test]
fn only_the_last_sixteen_positions_part_of_a_sweep_can_be_taken_back() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let mut plain =
        Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, CudaOps::new(&dev, &module).unwrap()).unwrap();
    let wide_ops = WideOps::new(CudaOps::new(&dev, &module).unwrap());
    let mut wide = Inkling::with_ops(load_config(&dir).unwrap(), &weights, 64, wide_ops).unwrap();
    let text = tokens(40);
    // 37 positions are 16, 16 and 5: the last part keeps its first and gives back up to 4.
    wide.feed(&text[..37], 0).unwrap();
    let e = wide.undo(5).unwrap_err();
    assert!(matches!(e, ModelError::Capacity(_)), "{e}");
    wide.feed(&text[..37], 0).unwrap();
    wide.undo(4).unwrap();
    // The model is then as if 33 positions had been fed: the next ones score the same.
    plain.feed(&text[..16], 0).unwrap();
    plain.feed(&text[16..32], 16).unwrap();
    plain.feed(&text[32..33], 32).unwrap();
    let (got, want) =
        (wide.forward_tokens(&text[33..40], 33).unwrap(), plain.forward_tokens(&text[33..40], 33).unwrap());
    assert!(got == want, "the scores after taking back 4 of a sweep of 37 differ");
}

#[test]
fn a_sweep_of_too_many_positions_is_refused() {
    let dir = tiny();
    let weights = Weights::open_dir(&dir).unwrap();
    let dev = CudaDevice::open(0).expect("GPU");
    let module = Module::load_embedded(&dev).unwrap();
    let wide_ops = WideOps::new(CudaOps::new(&dev, &module).unwrap());
    let room = MAX_SWEEP + 8;
    let mut wide = Inkling::with_ops(load_config(&dir).unwrap(), &weights, room, wide_ops).unwrap();
    let e = wide.feed(&tokens(MAX_SWEEP + 1), 0).unwrap_err();
    assert!(matches!(e, ModelError::Capacity(_)), "{e}");
    // And the model goes on as if it had not been asked.
    wide.feed(&tokens(20), 0).unwrap();
}
