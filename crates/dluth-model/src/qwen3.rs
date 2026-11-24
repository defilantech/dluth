//! The dense Qwen3 forward pass on the CPU, one token at a time, in `f32`.
//!
//! This is the reference model: slow, plain, and the thing every faster path is compared
//! with. It keeps a key-value cache so each new token costs one pass through the layers
//! instead of reprocessing the whole text.

use crate::config::{validate_weights, Qwen3Config};
use crate::error::ModelError;
use crate::weights::Weights;
use dluth_core::bf16_to_f32;
use dluth_kernels::cpu;

/// One transformer layer: its weights (matrices stay as stored BF16 bytes; the small norm
/// vectors are widened once) and its slice of the key-value cache.
struct Layer<'w> {
    input_norm: Vec<f32>,
    q_proj: &'w [u8],
    k_proj: &'w [u8],
    v_proj: &'w [u8],
    o_proj: &'w [u8],
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_norm: Vec<f32>,
    gate_proj: &'w [u8],
    up_proj: &'w [u8],
    down_proj: &'w [u8],
    /// Keys of every position seen so far: `[position][kv_head][head_dim]`.
    k_cache: Vec<f32>,
    /// Values, same layout.
    v_cache: Vec<f32>,
}

/// Buffers reused for every token, so the model itself allocates nothing per token. (The
/// attention reference op still makes one small scratch vector per call.)
struct Scratch {
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    head: Vec<f32>,
    attended: Vec<f32>,
    delta: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    gated: Vec<f32>,
}

pub struct Qwen3Cpu<'w> {
    pub cfg: Qwen3Config,
    embed: &'w [u8],
    layers: Vec<Layer<'w>>,
    final_norm: Vec<f32>,
    lm_head: &'w [u8],
    max_seq: usize,
    /// How many positions of the cache hold real keys and values.
    cached: usize,
    s: Scratch,
}

fn norm_weights(w: &Weights, name: &str) -> Result<Vec<f32>, ModelError> {
    Ok(w.bytes(name)?.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect())
}

/// Normalizes each head of `heads` separately with the same learned weights.
fn rmsnorm_per_head(heads: &mut [f32], weights: &[f32], eps: f32, tmp: &mut [f32]) {
    for head in heads.chunks_exact_mut(weights.len()) {
        cpu::rmsnorm(head, weights, eps, tmp);
        head.copy_from_slice(tmp);
    }
}

/// The residual connection: `x += delta`. Done in place because the hidden state is both an
/// input and the output; `cpu::vecadd` writes to a third buffer and cannot alias like this.
fn add_into(x: &mut [f32], delta: &[f32]) {
    for (a, b) in x.iter_mut().zip(delta) {
        *a += b;
    }
}

/// A zeroed cache for `positions * width` floats, or a refusal if that much memory cannot
/// be reserved. The position limit comes from a configuration file, so it is not trusted.
fn reserve_cache(positions: usize, width: usize) -> Result<Vec<f32>, ModelError> {
    let refusal = || ModelError::Capacity(format!("cannot reserve a cache of {positions} positions of {width} values"));
    let len = positions.checked_mul(width).ok_or_else(refusal)?;
    let mut cache = Vec::new();
    cache.try_reserve_exact(len).map_err(|_| refusal())?;
    cache.resize(len, 0.0);
    Ok(cache)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}

impl<'w> Qwen3Cpu<'w> {
    /// Checks the weights against the configuration and reserves a key-value cache for
    /// `max_seq` positions.
    pub fn new(cfg: Qwen3Config, w: &'w Weights, max_seq: usize) -> Result<Self, ModelError> {
        if max_seq == 0 || max_seq > cfg.max_position_embeddings {
            return Err(ModelError::Capacity(format!(
                "max_seq {max_seq} must be between 1 and the model's limit of {}",
                cfg.max_position_embeddings
            )));
        }
        validate_weights(&cfg, w)?;

        let kv_width = cfg.num_key_value_heads * cfg.head_dim;
        let q_width = cfg.num_attention_heads * cfg.head_dim;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_norm: norm_weights(w, &format!("{p}.input_layernorm.weight"))?,
                q_proj: w.bytes(&format!("{p}.self_attn.q_proj.weight"))?,
                k_proj: w.bytes(&format!("{p}.self_attn.k_proj.weight"))?,
                v_proj: w.bytes(&format!("{p}.self_attn.v_proj.weight"))?,
                o_proj: w.bytes(&format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: norm_weights(w, &format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: norm_weights(w, &format!("{p}.self_attn.k_norm.weight"))?,
                post_norm: norm_weights(w, &format!("{p}.post_attention_layernorm.weight"))?,
                gate_proj: w.bytes(&format!("{p}.mlp.gate_proj.weight"))?,
                up_proj: w.bytes(&format!("{p}.mlp.up_proj.weight"))?,
                down_proj: w.bytes(&format!("{p}.mlp.down_proj.weight"))?,
                k_cache: reserve_cache(max_seq, kv_width)?,
                v_cache: reserve_cache(max_seq, kv_width)?,
            });
        }
        let embed = w.bytes("model.embed_tokens.weight")?;
        let lm_head = if cfg.tie_word_embeddings { embed } else { w.bytes("lm_head.weight")? };
        let s = Scratch {
            x: vec![0.0; cfg.hidden_size],
            normed: vec![0.0; cfg.hidden_size],
            q: vec![0.0; q_width],
            k: vec![0.0; kv_width],
            v: vec![0.0; kv_width],
            head: vec![0.0; cfg.head_dim],
            attended: vec![0.0; q_width],
            delta: vec![0.0; cfg.hidden_size],
            gate: vec![0.0; cfg.intermediate_size],
            up: vec![0.0; cfg.intermediate_size],
            gated: vec![0.0; cfg.intermediate_size],
        };
        let final_norm = norm_weights(w, "model.norm.weight")?;
        Ok(Self { cfg, embed, layers, final_norm, lm_head, max_seq, cached: 0, s })
    }

    /// Runs one token through every layer, writing its keys and values into the cache at
    /// `pos`. Leaves the final hidden state in the scratch buffer.
    fn advance(&mut self, token: u32, pos: usize) -> Result<(), ModelError> {
        let cfg = &self.cfg;
        if token as usize >= cfg.vocab_size {
            return Err(ModelError::Capacity(format!(
                "token id {token} is outside the vocabulary of {}",
                cfg.vocab_size
            )));
        }
        if pos >= self.max_seq {
            return Err(ModelError::Capacity(format!(
                "position {pos} is outside the cache of {} positions",
                self.max_seq
            )));
        }
        if pos > self.cached {
            return Err(ModelError::Capacity(format!(
                "position {pos} skips ahead: only {} positions are cached",
                self.cached
            )));
        }
        let (hidden, inter, d) = (cfg.hidden_size, cfg.intermediate_size, cfg.head_dim);
        let (n_heads, n_kv, eps, theta) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.rms_norm_eps, cfg.rope_theta);
        let (q_width, kv_width) = (n_heads * d, n_kv * d);
        let s = &mut self.s;

        // Embedding: the token's row of the table.
        let row = &self.embed[token as usize * hidden * 2..(token as usize + 1) * hidden * 2];
        for (x, c) in s.x.iter_mut().zip(row.as_chunks::<2>().0) {
            *x = bf16_to_f32(u16::from_le_bytes(*c));
        }

        for layer in &mut self.layers {
            // Attention block.
            cpu::rmsnorm(&s.x, &layer.input_norm, eps, &mut s.normed);
            cpu::matvec_bf16(layer.q_proj, q_width, hidden, &s.normed, &mut s.q);
            cpu::matvec_bf16(layer.k_proj, kv_width, hidden, &s.normed, &mut s.k);
            cpu::matvec_bf16(layer.v_proj, kv_width, hidden, &s.normed, &mut s.v);
            // Qwen3 normalizes every query and key head before the position rotation.
            rmsnorm_per_head(&mut s.q, &layer.q_norm, eps, &mut s.head);
            rmsnorm_per_head(&mut s.k, &layer.k_norm, eps, &mut s.head);
            cpu::rope_inplace(&mut s.q, n_heads, d, pos, theta);
            cpu::rope_inplace(&mut s.k, n_kv, d, pos, theta);
            layer.k_cache[pos * kv_width..(pos + 1) * kv_width].copy_from_slice(&s.k);
            layer.v_cache[pos * kv_width..(pos + 1) * kv_width].copy_from_slice(&s.v);
            // The new token attends over every cached position, itself included.
            cpu::attention_one_query(&s.q, &layer.k_cache, &layer.v_cache, n_heads, n_kv, d, pos + 1, &mut s.attended);
            cpu::matvec_bf16(layer.o_proj, hidden, q_width, &s.attended, &mut s.delta);
            add_into(&mut s.x, &s.delta);

            // Feed-forward block.
            cpu::rmsnorm(&s.x, &layer.post_norm, eps, &mut s.normed);
            cpu::matvec_bf16(layer.gate_proj, inter, hidden, &s.normed, &mut s.gate);
            cpu::matvec_bf16(layer.up_proj, inter, hidden, &s.normed, &mut s.up);
            cpu::silu_mul(&s.gate, &s.up, &mut s.gated);
            cpu::matvec_bf16(layer.down_proj, hidden, inter, &s.gated, &mut s.delta);
            add_into(&mut s.x, &s.delta);
        }
        self.cached = pos + 1;
        Ok(())
    }

    /// Scores every token in the vocabulary from the current hidden state.
    fn logits(&mut self) -> Vec<f32> {
        let s = &mut self.s;
        cpu::rmsnorm(&s.x, &self.final_norm, self.cfg.rms_norm_eps, &mut s.normed);
        let mut logits = vec![0.0; self.cfg.vocab_size];
        cpu::matvec_bf16(self.lm_head, self.cfg.vocab_size, self.cfg.hidden_size, &s.normed, &mut logits);
        logits
    }

    /// Feeds one token at position `pos` and returns the logits for the token after it.
    /// Positions must be fed in order starting from 0; feeding an earlier position again
    /// rewinds the cache to that point.
    pub fn forward_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, ModelError> {
        self.advance(token, pos)?;
        Ok(self.logits())
    }

    /// Greedy generation: feed the prompt, then repeatedly take the highest-scoring token.
    /// Stops after `max_new` tokens or after producing `eos` (which is included).
    /// Refuses before computing anything if the prompt plus `max_new` cannot fit the cache.
    pub fn generate_greedy(&mut self, prompt_ids: &[u32], max_new: usize, eos: u32) -> Result<Vec<u32>, ModelError> {
        self.generate_greedy_with(prompt_ids, max_new, eos, |_| {})
    }

    /// As `generate_greedy`, and also calls `on_token` with each token the moment it is chosen,
    /// so a caller can show text as it appears and time the steps.
    pub fn generate_greedy_with(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        eos: u32,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>, ModelError> {
        if prompt_ids.is_empty() {
            return Err(ModelError::Capacity("empty prompt: generation needs at least one token".into()));
        }
        if prompt_ids.len() + max_new > self.max_seq {
            return Err(ModelError::Capacity(format!(
                "prompt of {} tokens plus {max_new} new tokens exceeds the cache of {} positions",
                prompt_ids.len(),
                self.max_seq
            )));
        }
        self.cached = 0;
        let mut pos = 0;
        // Only the last prompt token's logits are needed, so the output head is skipped before it.
        for &token in prompt_ids {
            self.advance(token, pos)?;
            pos += 1;
        }
        let mut generated = Vec::with_capacity(max_new);
        while generated.len() < max_new {
            let next = argmax(&self.logits());
            generated.push(next);
            on_token(next);
            if next == eos || generated.len() == max_new {
                break;
            }
            self.advance(next, pos)?;
            pos += 1;
        }
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_config;
    use crate::testutil::{TempDir, TinyModel};

    const NO_EOS: u32 = 9999;

    fn open(seed: u64) -> (TempDir, Qwen3Config, Weights) {
        let d = TinyModel::random(seed).write();
        let cfg = load_config(d.path()).unwrap();
        let w = Weights::open_dir(d.path()).unwrap();
        (d, cfg, w)
    }

    fn capacity_message<T: std::fmt::Debug>(r: Result<T, ModelError>) -> String {
        match r {
            Err(ModelError::Capacity(m)) => m,
            other => panic!("expected Capacity, got {other:?}"),
        }
    }

    #[test]
    fn tiny_model_is_deterministic() {
        let (_d, cfg, w) = open(42);
        let a = Qwen3Cpu::new(cfg.clone(), &w, 16).unwrap().generate_greedy(&[1, 2, 3], 5, NO_EOS).unwrap();
        let b = Qwen3Cpu::new(cfg, &w, 16).unwrap().generate_greedy(&[1, 2, 3], 5, NO_EOS).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
        assert!(a.iter().all(|&t| t < 10));
    }

    #[test]
    fn a_model_can_be_reused_for_a_second_generation() {
        let (_d, cfg, w) = open(42);
        let mut m = Qwen3Cpu::new(cfg, &w, 16).unwrap();
        let first = m.generate_greedy(&[1, 2, 3], 5, NO_EOS).unwrap();
        let _other = m.generate_greedy(&[4, 5, 6, 7], 3, NO_EOS).unwrap();
        assert_eq!(m.generate_greedy(&[1, 2, 3], 5, NO_EOS).unwrap(), first);
    }

    #[test]
    fn logits_have_vocab_length_and_are_finite() {
        let (_d, cfg, w) = open(3);
        let mut m = Qwen3Cpu::new(cfg, &w, 4).unwrap();
        let l = m.forward_token(1, 0).unwrap();
        assert_eq!(l.len(), 10);
        assert!(l.iter().all(|v| v.is_finite()));
        assert!(l.iter().any(|&v| v != l[0]), "logits should not all be equal");
    }

    #[test]
    fn earlier_tokens_change_later_logits() {
        let (_d, cfg, w) = open(5);
        let mut alone = Qwen3Cpu::new(cfg.clone(), &w, 4).unwrap();
        let mut after_one = Qwen3Cpu::new(cfg.clone(), &w, 4).unwrap();
        let mut after_other = Qwen3Cpu::new(cfg, &w, 4).unwrap();
        let base = alone.forward_token(2, 0).unwrap();
        after_one.forward_token(1, 0).unwrap();
        let with_history = after_one.forward_token(2, 1).unwrap();
        after_other.forward_token(7, 0).unwrap();
        let with_other_history = after_other.forward_token(2, 1).unwrap();
        assert_ne!(base, with_history);
        assert_ne!(with_history, with_other_history);
    }

    #[test]
    fn greedy_generation_follows_the_largest_logit() {
        let (_d, cfg, w) = open(11);
        let mut m = Qwen3Cpu::new(cfg.clone(), &w, 8).unwrap();
        let generated = m.generate_greedy(&[3, 1], 2, NO_EOS).unwrap();
        let mut step = Qwen3Cpu::new(cfg, &w, 8).unwrap();
        step.forward_token(3, 0).unwrap();
        let l = step.forward_token(1, 1).unwrap();
        let best = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
        assert_eq!(generated[0], best);
        let l = step.forward_token(best, 2).unwrap();
        let best = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
        assert_eq!(generated[1], best);
    }

    #[test]
    fn each_token_is_reported_as_it_is_produced() {
        let (_d, cfg, w) = open(42);
        let mut seen = Vec::new();
        let all =
            Qwen3Cpu::new(cfg, &w, 16).unwrap().generate_greedy_with(&[1, 2, 3], 5, NO_EOS, |t| seen.push(t)).unwrap();
        assert_eq!(seen, all);
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn stops_at_eos() {
        let (_d, cfg, w) = open(7);
        let first = Qwen3Cpu::new(cfg.clone(), &w, 16).unwrap().generate_greedy(&[1], 1, NO_EOS).unwrap()[0];
        // Declaring the first generated token to be the end token stops generation right there.
        assert_eq!(Qwen3Cpu::new(cfg, &w, 16).unwrap().generate_greedy(&[1], 8, first).unwrap(), vec![first]);
    }

    #[test]
    fn refuses_when_prompt_plus_generation_exceeds_kv() {
        let (_d, cfg, w) = open(1);
        let mut m = Qwen3Cpu::new(cfg, &w, 8).unwrap();
        let s = capacity_message(m.generate_greedy(&[1, 2, 3, 4, 5], 4, NO_EOS));
        assert!(s.contains('5') && s.contains('4') && s.contains('8'), "{s}");
        // Exactly filling the cache is allowed.
        assert_eq!(m.generate_greedy(&[1, 2, 3, 4, 5], 3, NO_EOS).unwrap().len(), 3);
    }

    #[test]
    fn refuses_an_empty_prompt() {
        let (_d, cfg, w) = open(1);
        let mut m = Qwen3Cpu::new(cfg, &w, 8).unwrap();
        assert!(capacity_message(m.generate_greedy(&[], 4, NO_EOS)).contains("empty"));
    }

    #[test]
    fn forward_token_refuses_out_of_range_token_and_position() {
        let (_d, cfg, w) = open(1);
        let mut m = Qwen3Cpu::new(cfg, &w, 4).unwrap();
        let s = capacity_message(m.forward_token(10, 0));
        assert!(s.contains("10") && s.contains("token"), "{s}");
        let s = capacity_message(m.forward_token(1, 4));
        assert!(s.contains('4') && s.contains("position"), "{s}");
        // Skipping ahead would attend over cache slots that were never written.
        let s = capacity_message(m.forward_token(1, 2));
        assert!(s.contains('2') && s.contains("position"), "{s}");
        m.forward_token(1, 0).unwrap();
        m.forward_token(1, 1).unwrap();
    }

    #[test]
    fn new_refuses_max_seq_of_zero_or_beyond_model_limit() {
        let (_d, cfg, w) = open(1);
        let s = capacity_message(Qwen3Cpu::new(cfg.clone(), &w, 0).map(|_| ()));
        assert!(s.contains('0') && s.contains("64"), "{s}");
        let s = capacity_message(Qwen3Cpu::new(cfg.clone(), &w, 65).map(|_| ()));
        assert!(s.contains("65") && s.contains("64"), "{s}");
        assert!(Qwen3Cpu::new(cfg, &w, 64).is_ok());
    }

    #[test]
    fn new_refuses_a_cache_that_cannot_be_reserved_instead_of_aborting() {
        // A hand-written configuration can claim any position limit; the weights stay tiny.
        let d = TinyModel::random(1).with_config(|c| c["max_position_embeddings"] = (1u64 << 60).into()).write();
        let cfg = load_config(d.path()).unwrap();
        let w = Weights::open_dir(d.path()).unwrap();
        for max_seq in [1usize << 50, 1usize << 60] {
            let s = capacity_message(Qwen3Cpu::new(cfg.clone(), &w, max_seq).map(|_| ()));
            assert!(s.contains("cache") && s.contains(&max_seq.to_string()), "{s}");
        }
    }

    #[test]
    fn new_validates_the_weights() {
        let d = TinyModel::random(1).without("model.layers.0.self_attn.q_norm.weight").write();
        let cfg = load_config(d.path()).unwrap();
        let w = Weights::open_dir(d.path()).unwrap();
        assert!(matches!(Qwen3Cpu::new(cfg, &w, 4), Err(ModelError::MissingTensor(n)) if n.ends_with("q_norm.weight")));
    }

    #[test]
    fn tied_embeddings_reuse_the_embedding_table_as_the_output_head() {
        let tied = TinyModel::random(9)
            .without("lm_head.weight")
            .with_config(|c| c["tie_word_embeddings"] = true.into())
            .write();
        let cfg = load_config(tied.path()).unwrap();
        let w = Weights::open_dir(tied.path()).unwrap();
        let l = Qwen3Cpu::new(cfg, &w, 4).unwrap().forward_token(1, 0).unwrap();
        assert_eq!(l.len(), 10);
        assert!(l.iter().all(|v| v.is_finite()));
    }
}
