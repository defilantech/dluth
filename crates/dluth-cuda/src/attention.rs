//! The attention sublayer's operations other than its matrix products, on the device.
//!
//! Every function starts its kernel and returns without waiting. Each checks that
//! everything the kernel will read or write lies inside the buffers it was given.

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::floats::Floats;
use crate::module::{Kernel, Module};
use std::ffi::c_void;

fn refuse<T>(what: &str, why: String) -> Result<T, CudaError> {
    Err(CudaError::Refused(format!("{what}: {why}")))
}

fn blocks(threads: u64, per_block: u32) -> Result<u32, CudaError> {
    u32::try_from(threads.div_ceil(per_block as u64))
        .map_err(|_| CudaError::Refused(format!("a grid of {threads} threads is too large")))
}

fn ptr(p: &mut u64) -> *mut c_void {
    p as *mut u64 as *mut c_void
}
fn num(p: &mut u32) -> *mut c_void {
    p as *mut u32 as *mut c_void
}
fn big(p: &mut u64) -> *mut c_void {
    p as *mut u64 as *mut c_void
}
fn real(p: &mut f32) -> *mut c_void {
    p as *mut f32 as *mut c_void
}

/// The shape of a layer's attention: query heads, key-value heads, and the size of a head.
#[derive(Clone, Copy, Debug)]
pub struct Heads {
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
}

/// The keys and values of the tokens a layer pass can still see: two buffers of `cap` rows
/// of `kv_heads * dim` floats. The token at position p is in row `p % cap`.
#[derive(Clone, Copy, Debug)]
pub struct History<'b> {
    pub keys: Floats<'b>,
    pub values: Floats<'b>,
    pub cap: usize,
}

/// The relative-position term: one vector of `rel_dim` per query head, and a table of
/// `rel_dim` rows of `reach` columns, one column per distance.
#[derive(Clone, Copy, Debug)]
pub struct Relative<'b> {
    pub rel: Floats<'b>,
    pub table: Floats<'b>,
    pub rel_dim: usize,
    pub reach: usize,
}

/// The attention kernels of a loaded module.
pub struct Attention<'m> {
    dev: &'m CudaDevice,
    scale_clip: Kernel<'m>,
    rmsnorm_plain: Kernel<'m>,
    sconv_fixed: Kernel<'m>,
    sconv_dynamic: Kernel<'m>,
    scores: Kernel<'m>,
    softmax: Kernel<'m>,
    values: Kernel<'m>,
    sigmoid_mul: Kernel<'m>,
}

impl<'m> Attention<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        Ok(Attention {
            dev,
            scale_clip: module.kernel("head_scale_clip")?,
            rmsnorm_plain: module.kernel("head_rmsnorm_plain")?,
            sconv_fixed: module.kernel("sconv_fixed")?,
            sconv_dynamic: module.kernel("sconv_dynamic")?,
            scores: module.kernel("attn_scores")?,
            softmax: module.kernel("attn_softmax")?,
            values: module.kernel("attn_values")?,
            sigmoid_mul: module.kernel("sigmoid_mul")?,
        })
    }

    /// In place, for each of `heads` vectors of `dim` in `x`: multiply by `scale` element by
    /// element (if given), shorten to length `clip` if longer, multiply by `late`.
    pub fn head_scale_clip(
        &self,
        x: Floats,
        scale: Option<Floats>,
        heads: usize,
        dim: usize,
        clip: f32,
        late: f32,
    ) -> Result<(), CudaError> {
        let what = "head_scale_clip";
        if heads == 0 || dim == 0 || heads > 1 << 20 || dim > 1 << 20 {
            return refuse(what, format!("{heads} heads of {dim}"));
        }
        // A value that is not a number is not finite, so it is refused here too.
        if clip <= 0.0 || !clip.is_finite() || !late.is_finite() {
            return refuse(what, format!("a clip of {clip} and a factor of {late}"));
        }
        x.check(self.dev, what, heads * dim)?;
        if let Some(s) = &scale {
            s.check(self.dev, what, heads * dim)?;
            if s.overlaps(&x) {
                return refuse(what, "the scale overlaps the vectors".into());
            }
        }
        let (mut px, mut ps) = (x.address(), scale.map_or(0, |s| s.address()));
        let (mut h, mut d, mut c, mut l) = (heads as u32, dim as u32, clip, late);
        let mut args = [ptr(&mut px), ptr(&mut ps), num(&mut h), num(&mut d), real(&mut c), real(&mut l)];
        // SAFETY: the arguments match head_scale_clip(float*, const float*, unsigned,
        // unsigned, float, float). The kernel reads and writes heads * dim floats of `x`
        // and reads as many of `scale` unless it is null; both were checked.
        unsafe { self.scale_clip.launch((blocks(heads as u64, 64)?, 1, 1), (64, 1, 1), &mut args) }
    }

    /// For each of `heads` vectors of `dim` in `x`: `out = x / sqrt(mean(x^2) + eps)`.
    pub fn head_rmsnorm_plain(
        &self,
        x: Floats,
        out: Floats,
        heads: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let what = "head_rmsnorm_plain";
        if heads == 0 || dim == 0 || heads > 1 << 20 || dim > 1 << 20 || eps < 0.0 || !eps.is_finite() {
            return refuse(what, format!("{heads} heads of {dim}, eps {eps}"));
        }
        x.check(self.dev, what, heads * dim)?;
        out.check(self.dev, what, heads * dim)?;
        if x.overlaps(&out) {
            return refuse(what, "the output overlaps the input".into());
        }
        let (mut px, mut po, mut h, mut d, mut e) = (x.address(), out.address(), heads as u32, dim as u32, eps);
        let mut args = [ptr(&mut px), ptr(&mut po), num(&mut h), num(&mut d), real(&mut e)];
        // SAFETY: the arguments match head_rmsnorm_plain(const float*, float*, unsigned,
        // unsigned, float); it reads heads * dim floats of `x` and writes as many to `out`.
        unsafe { self.rmsnorm_plain.launch((blocks(heads as u64, 64)?, 1, 1), (64, 1, 1), &mut args) }
    }

    /// The short convolution of the token at `pos` with fixed taps (`channels * n` of them),
    /// written to `out`; the token then takes its place in `ring`, which holds the last
    /// `n - 1` tokens of `channels` floats. Tokens must be given in order of position.
    pub fn sconv_fixed(
        &self,
        current: Floats,
        ring: Floats,
        taps: Floats,
        out: Floats,
        n: usize,
        pos: usize,
    ) -> Result<(), CudaError> {
        let what = "sconv_fixed";
        let channels = current.len;
        self.conv_checks(what, current, ring, out, n)?;
        taps.check(self.dev, what, channels * n)?;
        let (mut pc, mut pr, mut pt, mut po) = (current.address(), ring.address(), taps.address(), out.address());
        let (mut c, mut taps_n, mut p, mut norm) =
            (channels as u32, n as u32, pos as u64, 1.0 / ((n + 1) as f32).sqrt());
        let mut args = [
            ptr(&mut pc),
            ptr(&mut pr),
            ptr(&mut pt),
            ptr(&mut po),
            num(&mut c),
            num(&mut taps_n),
            big(&mut p),
            real(&mut norm),
        ];
        // SAFETY: the arguments match sconv_fixed(const float*, float*, const float*, float*,
        // unsigned, unsigned, unsigned long long, float). It reads `channels` floats of
        // `current`, channels * n of `taps`, reads and writes (n - 1) * channels of `ring`
        // and writes `channels` to `out`; all checked, and `out` overlaps none of them.
        unsafe { self.sconv_fixed.launch((blocks(channels as u64, 256)?, 1, 1), (256, 1, 1), &mut args) }
    }

    /// The short convolution whose taps are `fixed_taps` (`n * channels`) plus `token_taps`
    /// (`heads * n`, computed for this token and shared by the channels of a head).
    #[allow(clippy::too_many_arguments)]
    pub fn sconv_dynamic(
        &self,
        current: Floats,
        ring: Floats,
        fixed_taps: Floats,
        token_taps: Floats,
        out: Floats,
        n: usize,
        heads: usize,
        pos: usize,
    ) -> Result<(), CudaError> {
        let what = "sconv_dynamic";
        let channels = current.len;
        self.conv_checks(what, current, ring, out, n)?;
        if heads == 0 || !channels.is_multiple_of(heads) {
            return refuse(what, format!("{channels} channels in {heads} heads"));
        }
        fixed_taps.check(self.dev, what, channels * n)?;
        token_taps.check(self.dev, what, heads * n)?;
        if token_taps.overlaps(&out) || token_taps.overlaps(&ring) {
            return refuse(what, "the token taps overlap something written".into());
        }
        let (mut pc, mut pr, mut pf, mut pt, mut po) =
            (current.address(), ring.address(), fixed_taps.address(), token_taps.address(), out.address());
        let (mut c, mut taps_n, mut h, mut p) = (channels as u32, n as u32, heads as u32, pos as u64);
        let mut norm = 1.0 / ((2 * n + 1) as f32).sqrt();
        let mut args = [
            ptr(&mut pc),
            ptr(&mut pr),
            ptr(&mut pf),
            ptr(&mut pt),
            ptr(&mut po),
            num(&mut c),
            num(&mut taps_n),
            num(&mut h),
            big(&mut p),
            real(&mut norm),
        ];
        // SAFETY: the arguments match sconv_dynamic(const float*, float*, const float*,
        // const float*, float*, unsigned, unsigned, unsigned, unsigned long long, float).
        // Extents as for sconv_fixed, plus heads * n floats of `token_taps`; all checked.
        unsafe { self.sconv_dynamic.launch((blocks(channels as u64, 256)?, 1, 1), (256, 1, 1), &mut args) }
    }

    fn conv_checks(&self, what: &str, current: Floats, ring: Floats, out: Floats, n: usize) -> Result<(), CudaError> {
        let channels = current.len;
        if n == 0 || n > 64 || channels == 0 || channels > 1 << 24 {
            return refuse(what, format!("{n} taps for {channels} channels"));
        }
        current.check(self.dev, what, channels)?;
        out.check(self.dev, what, channels)?;
        if n > 1 {
            ring.check(self.dev, what, (n - 1) * channels)?;
            if ring.overlaps(&current) || ring.overlaps(&out) {
                return refuse(what, "the ring overlaps the token or the output".into());
            }
        }
        if out.overlaps(&current) {
            return refuse(what, "the output overlaps the token".into());
        }
        Ok(())
    }

    /// Attention for the query token at position `pos`, whose own key and value are already
    /// in the history: `out[h] = sum over the seen tokens of weight * value`, the weights
    /// being a softmax over the seen tokens' scores and the head's sink score.
    ///
    /// `seen` is how many tokens, the newest included, the query may look at: at most
    /// `pos + 1` and at most `history.cap`. `weights` is scratch of `heads * (seen + 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn attend(
        &self,
        shape: Heads,
        q: Floats,
        history: History,
        relative: Relative,
        sinks: Floats,
        weights: Floats,
        out: Floats,
        pos: usize,
        seen: usize,
    ) -> Result<(), CudaError> {
        let what = "attend";
        let Heads { heads, kv_heads, dim } = shape;
        let Relative { rel, table, rel_dim, reach } = relative;
        if heads == 0
            || kv_heads == 0
            || dim == 0
            || !heads.is_multiple_of(kv_heads)
            || heads > 1 << 16
            || dim > 1 << 16
        {
            return refuse(what, format!("{heads} heads over {kv_heads} key-value heads of {dim}"));
        }
        if rel_dim == 0 || reach == 0 || rel_dim > 1 << 16 || reach > u32::MAX as usize {
            return refuse(what, format!("a relative-position table of {rel_dim} by {reach}"));
        }
        if seen == 0 || seen > history.cap || seen - 1 > pos {
            return refuse(what, format!("{seen} tokens seen at position {pos} with a history of {}", history.cap));
        }
        let width = kv_heads * dim;
        q.check(self.dev, what, heads * dim)?;
        out.check(self.dev, what, heads * dim)?;
        history.keys.check(self.dev, what, history.cap * width)?;
        history.values.check(self.dev, what, history.cap * width)?;
        rel.check(self.dev, what, heads * rel_dim)?;
        table.check(self.dev, what, rel_dim * reach)?;
        sinks.check(self.dev, what, heads)?;
        weights.check(self.dev, what, heads * (seen + 1))?;
        for read in [&q, &history.keys, &history.values, &rel, &table, &sinks] {
            if weights.overlaps(read) || out.overlaps(read) {
                return refuse(what, "the scratch or the output overlaps an input".into());
            }
        }
        if weights.overlaps(&out) {
            return refuse(what, "the scratch overlaps the output".into());
        }

        let (mut pq, mut pk, mut pv, mut pr, mut pt, mut ps, mut pw, mut po) = (
            q.address(),
            history.keys.address(),
            history.values.address(),
            rel.address(),
            table.address(),
            sinks.address(),
            weights.address(),
            out.address(),
        );
        let (mut h, mut kv, mut d, mut rd, mut rc) =
            (heads as u32, kv_heads as u32, dim as u32, rel_dim as u32, reach as u32);
        let (mut p, mut n, mut cap, mut count) = (pos as u64, seen as u64, history.cap as u64, seen as u64 + 1);
        let mut scale = 1.0 / (dim as f32).sqrt();

        let mut args = [
            ptr(&mut pq),
            ptr(&mut pk),
            ptr(&mut pr),
            ptr(&mut pt),
            ptr(&mut ps),
            ptr(&mut pw),
            num(&mut h),
            num(&mut kv),
            num(&mut d),
            num(&mut rd),
            num(&mut rc),
            real(&mut scale),
            big(&mut p),
            big(&mut n),
            big(&mut cap),
        ];
        // SAFETY (the three launches): each argument list matches its kernel's parameters in
        // order and type. attn_scores reads the query, `seen` rows of the keys (rows
        // (pos - d) % cap for d < seen, all below cap), the relative vectors, columns below
        // `reach` of the table and the sinks, and writes heads * (seen + 1) weights.
        // attn_softmax rewrites those weights in place, one block of at most 1024 threads
        // per head. attn_values reads them and `seen` rows of the values and writes
        // heads * dim floats to `out`. Every extent was checked above, and nothing written
        // overlaps anything read.
        unsafe { self.scores.launch((blocks(heads as u64 * count, 256)?, 1, 1), (256, 1, 1), &mut args)? };
        let mut args = [ptr(&mut pw), num(&mut h), big(&mut count)];
        unsafe { self.softmax.launch((heads as u32, 1, 1), (256, 1, 1), &mut args)? };
        let mut args = [
            ptr(&mut pw),
            ptr(&mut pv),
            ptr(&mut po),
            num(&mut h),
            num(&mut kv),
            num(&mut d),
            big(&mut p),
            big(&mut n),
            big(&mut cap),
        ];
        unsafe { self.values.launch((blocks((heads * dim) as u64, 256)?, 1, 1), (256, 1, 1), &mut args) }
    }

    /// Softmax in place over each of `heads` runs of `count` scores in `weights`.
    pub fn softmax(&self, weights: Floats, heads: usize, count: usize) -> Result<(), CudaError> {
        let what = "attn_softmax";
        if heads == 0 || count == 0 || heads > 1 << 16 {
            return refuse(what, format!("{heads} runs of {count}"));
        }
        weights.check(self.dev, what, heads * count)?;
        let (mut pw, mut h, mut n) = (weights.address(), heads as u32, count as u64);
        let mut args = [ptr(&mut pw), num(&mut h), big(&mut n)];
        // SAFETY: the arguments match attn_softmax(float*, unsigned, unsigned long long);
        // one block of 256 threads per run reads and writes that run's `count` floats.
        unsafe { self.softmax.launch((heads as u32, 1, 1), (256, 1, 1), &mut args) }
    }

    /// In place: `x[i] *= sigmoid(gate[i])`.
    pub fn sigmoid_mul(&self, x: Floats, gate: Floats) -> Result<(), CudaError> {
        let what = "sigmoid_mul";
        let n = x.len;
        if n > u32::MAX as usize {
            return refuse(what, format!("{n} values"));
        }
        x.check(self.dev, what, n)?;
        gate.check(self.dev, what, n)?;
        if x.overlaps(&gate) {
            return refuse(what, "the gate overlaps the values".into());
        }
        let (mut px, mut pg, mut count) = (x.address(), gate.address(), n as u32);
        let mut args = [ptr(&mut px), ptr(&mut pg), num(&mut count)];
        // SAFETY: the arguments match sigmoid_mul(float*, const float*, unsigned); it reads
        // and writes n floats of `x` and reads n of `gate`.
        unsafe { self.sigmoid_mul.launch((blocks(n as u64, 256)?, 1, 1), (256, 1, 1), &mut args) }
    }
}
