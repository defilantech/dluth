//! Sweeps of more positions than one kernel launch takes.
//!
//! The device operations (`CudaOps`) take up to `MAX_BATCH` positions at a time. Reading a
//! prompt that way costs most where the model's experts are: each position chooses a few of
//! a layer's experts, a sweep of sixteen positions chooses nearly as many different experts
//! as it has choices, and every one of them is read from memory for one or two vectors. A
//! layer has only so many experts, though: with more positions in a sweep the same experts
//! are chosen again, and an expert is read once for every position that chose it.
//!
//! `WideOps` is the operations for such sweeps. A value is the sweep cut into parts of up
//! to `MAX_BATCH` positions. Every operation but one is `CudaOps`'s, part after part, in
//! order, which is what feeding the parts as batches of their own would do. The expert step
//! is done for the whole sweep at once. The result is what feeding the parts one after
//! another gives, to the last bit: an expert's product for a vector does not depend on
//! which other vectors are in its job.
//!
//! Taking tokens back (`undo`) reaches into the last part of the last sweep only, which is
//! all that generation with guesses needs: its sweeps are a part or less.

use crate::attention::MAX_BATCH;
use crate::ops::{CudaOps, DeviceBlend, DevicePass, DeviceVec, OpsStats};
use dluth_core::Buffer;
use dluth_model::inkling::{Gate, InklingConfig, Layer, Ops};
use dluth_model::ModelError;
use std::cell::RefCell;
use std::rc::Rc;

/// The most positions in one sweep: what the expert step's kernels take.
pub const MAX_SWEEP: usize = 1024;

/// A sweep's vectors: its positions in order, in parts of up to `MAX_BATCH`, every part but
/// the last full.
#[derive(Clone)]
pub struct WideVec(Vec<DeviceVec>);

impl WideVec {
    /// The parts, in order.
    pub fn parts(&self) -> &[DeviceVec] {
        &self.0
    }

    /// How many positions it holds.
    pub fn count(&self) -> usize {
        self.0.iter().map(DeviceVec::count).sum()
    }
}

/// What one application of a layer remembers, with where the last sweep started and how
/// many positions its last part has left.
pub struct WidePass {
    inner: DevicePass,
    pos: usize,
    last_part: usize,
}

/// The device operations for sweeps of up to `MAX_SWEEP` positions.
pub struct WideOps<'m> {
    inner: CudaOps<'m>,
}

fn unequal(what: &str, a: &WideVec, b: &WideVec) -> ModelError {
    ModelError::Capacity(format!("{what}: sweeps of {} and {} positions", a.count(), b.count()))
}

/// Refuses two values that are not cut into the same parts.
fn same_parts(what: &str, a: &WideVec, b: &WideVec) -> Result<(), ModelError> {
    if a.0.len() == b.0.len() && a.0.iter().zip(&b.0).all(|(x, y)| x.count() == y.count()) {
        Ok(())
    } else {
        Err(unequal(what, a, b))
    }
}

impl<'m> WideOps<'m> {
    pub fn new(inner: CudaOps<'m>) -> Self {
        WideOps { inner }
    }

    /// See `CudaOps::stats`.
    pub fn stats(&self) -> Rc<RefCell<OpsStats>> {
        self.inner.stats()
    }

    /// See `CudaOps::device_weight`.
    pub fn device_weight(&mut self, bytes: &[u8]) -> Result<Buffer, ModelError> {
        self.inner.device_weight(bytes)
    }

    /// See `CudaOps::weight_bytes`.
    pub fn weight_bytes(&self) -> usize {
        self.inner.weight_bytes()
    }

    fn each(
        &mut self,
        x: &WideVec,
        mut op: impl FnMut(&mut CudaOps<'m>, &DeviceVec) -> Result<DeviceVec, ModelError>,
    ) -> Result<WideVec, ModelError> {
        x.0.iter().map(|part| op(&mut self.inner, part)).collect::<Result<_, _>>().map(WideVec)
    }
}

impl<'m> Ops for WideOps<'m> {
    type V = WideVec;
    type Blend = Vec<DeviceBlend>;
    type Pass = WidePass;

    fn new_pass(&mut self, cfg: &InklingConfig, local: bool, max_seq: usize) -> Result<WidePass, ModelError> {
        Ok(WidePass { inner: self.inner.new_pass(cfg, local, max_seq)?, pos: 0, last_part: 0 })
    }

    fn reset_pass(&mut self, pass: &mut WidePass) -> Result<(), ModelError> {
        (pass.pos, pass.last_part) = (0, 0);
        self.inner.reset_pass(&mut pass.inner)
    }

    fn undo(&mut self, pass: &mut WidePass, undone: usize) -> Result<(), ModelError> {
        // What `CudaOps::undo` rests on holds for the last part of a sweep, which was its
        // last batch. Positions of the parts before it are out of its reach.
        if undone > 0 && undone >= pass.last_part {
            return Err(ModelError::Capacity(format!(
                "{undone} tokens cannot be undone: the last {MAX_BATCH} positions of a sweep can be, less the first of them, and {} are left of those",
                pass.last_part
            )));
        }
        self.inner.undo(&mut pass.inner, undone)?;
        pass.last_part -= undone;
        Ok(())
    }

    fn count(&self, x: &WideVec) -> usize {
        x.count()
    }

    fn row(&mut self, x: &WideVec, index: usize) -> Result<WideVec, ModelError> {
        let mut first = 0;
        for part in &x.0 {
            if index < first + part.count() {
                return Ok(WideVec(vec![self.inner.row(part, index - first)?]));
            }
            first += part.count();
        }
        Err(ModelError::Capacity(format!("position {index} of a sweep of {first}")))
    }

    fn embed(
        &mut self,
        cfg: &InklingConfig,
        table: &[u8],
        norm: &[f32],
        tokens: &[u32],
    ) -> Result<WideVec, ModelError> {
        // Refused here, where a sweep enters the forward pass, before anything is computed.
        if tokens.is_empty() || tokens.len() > MAX_SWEEP {
            return Err(ModelError::Capacity(format!(
                "a sweep of {} tokens: the device operations take 1 to {MAX_SWEEP} positions at a time",
                tokens.len()
            )));
        }
        tokens
            .chunks(MAX_BATCH)
            .map(|part| self.inner.embed(cfg, table, norm, part))
            .collect::<Result<_, _>>()
            .map(WideVec)
    }

    fn normalize(&mut self, candidates: &[&WideVec]) -> Result<Vec<DeviceBlend>, ModelError> {
        let Some(first) = candidates.first() else {
            return Err(ModelError::Device("a blend of no candidates".into()));
        };
        for other in candidates {
            same_parts("a blend", first, other)?;
        }
        (0..first.0.len())
            .map(|part| {
                let of_part: Vec<&DeviceVec> = candidates.iter().map(|c| &c.0[part]).collect();
                self.inner.normalize(&of_part)
            })
            .collect()
    }

    fn mix(&mut self, blend: &Vec<DeviceBlend>, query: &[f32]) -> Result<WideVec, ModelError> {
        blend.iter().map(|part| self.inner.mix(part, query)).collect::<Result<_, _>>().map(WideVec)
    }

    fn mix_many(
        &mut self,
        blend: &Vec<DeviceBlend>,
        queries: &[&[f32]],
        norms: &[Option<&[f32]>],
        eps: f32,
    ) -> Result<Vec<WideVec>, ModelError> {
        // One value per query, each with every part.
        let mut out: Vec<WideVec> = queries.iter().map(|_| WideVec(Vec::with_capacity(blend.len()))).collect();
        for part in blend {
            let mixed = self.inner.mix_many(part, queries, norms, eps)?;
            if mixed.len() != out.len() {
                return Err(ModelError::Device(format!("{} blends for {} queries", mixed.len(), out.len())));
            }
            for (value, of_part) in out.iter_mut().zip(mixed) {
                value.0.push(of_part);
            }
        }
        Ok(out)
    }

    fn merge(
        &mut self,
        cfg: &InklingConfig,
        gate: &Gate,
        x: &WideVec,
        message: &WideVec,
    ) -> Result<WideVec, ModelError> {
        same_parts("a merge", x, message)?;
        let mut parts = message.0.iter();
        self.each(x, |ops, part| ops.merge(cfg, gate, part, parts.next().expect("the same parts")))
    }

    fn rmsnorm(&mut self, x: &WideVec, weights: &[f32], eps: f32) -> Result<WideVec, ModelError> {
        self.each(x, |ops, part| ops.rmsnorm(part, weights, eps))
    }

    fn attention(
        &mut self,
        cfg: &InklingConfig,
        layer: &Layer,
        pass: &mut WidePass,
        inputs: [&WideVec; 3],
        pos: usize,
    ) -> Result<WideVec, ModelError> {
        let [q, k, v] = inputs;
        same_parts("attention", q, k)?;
        same_parts("attention", q, v)?;
        // Part after part: a later part's positions see the earlier parts' in the history.
        let mut out = Vec::with_capacity(q.0.len());
        let mut at = pos;
        for ((q, k), v) in q.0.iter().zip(&k.0).zip(&v.0) {
            out.push(self.inner.attention(cfg, layer, &mut pass.inner, [q, k, v], at)?);
            at += q.count();
        }
        (pass.pos, pass.last_part) = (pos, q.0.last().map_or(0, DeviceVec::count));
        Ok(WideVec(out))
    }

    fn attn_conv(&mut self, layer: &Layer, pass: &mut WidePass, x: &WideVec) -> Result<WideVec, ModelError> {
        let mut at = pass.pos;
        self.each(x, |ops, part| {
            pass.inner.starts_at(at);
            at += part.count();
            ops.attn_conv(layer, &mut pass.inner, part)
        })
    }

    fn expert_conv(&mut self, layer: &Layer, pass: &mut WidePass, x: &WideVec) -> Result<WideVec, ModelError> {
        let mut at = pass.pos;
        self.each(x, |ops, part| {
            pass.inner.starts_at(at);
            at += part.count();
            ops.expert_conv(layer, &mut pass.inner, part)
        })
    }

    fn add(&mut self, a: &WideVec, b: &WideVec) -> Result<WideVec, ModelError> {
        same_parts("a sum", a, b)?;
        let mut parts = b.0.iter();
        self.each(a, |ops, part| ops.add(part, parts.next().expect("the same parts")))
    }

    fn experts(&mut self, cfg: &InklingConfig, layer: &Layer, x: &WideVec) -> Result<WideVec, ModelError> {
        self.inner.experts_of_parts(cfg, layer, &x.0).map(WideVec)
    }

    fn scores(&mut self, cfg: &InklingConfig, unembed: &[u8], x: &WideVec) -> Result<Vec<f32>, ModelError> {
        let mut all = Vec::new();
        for part in &x.0 {
            all.extend(self.inner.scores(cfg, unembed, part)?);
        }
        Ok(all)
    }
}
