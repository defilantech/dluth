//! The attention sublayer's operations other than its matrix products, on the device, for
//! a batch of consecutive tokens.
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

/// Most consecutive tokens one launch takes.
pub const MAX_BATCH: usize = 16;

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

/// The relative-position term: one vector of `rel_dim` per query token and head, and a
/// table of `rel_dim` rows of `reach` columns, one column per distance.
#[derive(Clone, Copy, Debug)]
pub struct Relative<'b> {
    pub rel: Floats<'b>,
    pub table: Floats<'b>,
    pub rel_dim: usize,
    pub reach: usize,
}

/// The factor a head is multiplied by late in a long context:
/// `1 + alpha * ln(max((position + 1) / floor, 1))`, where `pos` is the position of the
/// first token of the batch. With `alpha` 0 it is exactly 1.
#[derive(Clone, Copy, Debug)]
pub struct Late {
    pub pos: usize,
    pub floor: f32,
    pub alpha: f32,
}

impl Late {
    /// No factor.
    pub fn none() -> Self {
        Late { pos: 0, floor: 1.0, alpha: 0.0 }
    }
}

/// The attention kernels of a loaded module.
pub struct Attention<'m> {
    dev: &'m CudaDevice,
    scale_clip: Kernel<'m>,
    rmsnorm_plain: Kernel<'m>,
    sconv_fixed: Kernel<'m>,
    sconv_dynamic: Kernel<'m>,
    ring_store: Kernel<'m>,
    store_rows: Kernel<'m>,
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
            ring_store: module.kernel("ring_store")?,
            store_rows: module.kernel("store_rows")?,
            scores: module.kernel("attn_scores")?,
            softmax: module.kernel("attn_softmax")?,
            values: module.kernel("attn_values")?,
            sigmoid_mul: module.kernel("sigmoid_mul")?,
        })
    }

    /// In place, for each vector of `dim` in `x`: multiply by `scale` element by element (if
    /// given), shorten to length `clip` if longer, multiply by the late factor of its
    /// position. `x` holds `heads` vectors for each position of a batch; `scale` holds
    /// `heads` vectors and is used for every position.
    pub fn head_scale_clip(
        &self,
        x: Floats,
        scale: Option<Floats>,
        heads: usize,
        dim: usize,
        clip: f32,
        late: Late,
    ) -> Result<(), CudaError> {
        let what = "head_scale_clip";
        if heads == 0 || dim == 0 || heads > 1 << 20 || dim > 1 << 20 || !x.len.is_multiple_of(heads * dim) {
            return refuse(what, format!("{} floats for heads of {dim}, {heads} per position", x.len));
        }
        let vectors = x.len / dim;
        // A value that is not a number is not finite, so it is refused here too.
        if clip <= 0.0 || !clip.is_finite() || late.floor <= 0.0 || !late.floor.is_finite() || !late.alpha.is_finite() {
            return refuse(what, format!("a clip of {clip} and a late factor of {late:?}"));
        }
        if vectors == 0 || vectors > u32::MAX as usize {
            return refuse(what, format!("{vectors} vectors"));
        }
        x.check(self.dev, what, vectors * dim)?;
        if let Some(s) = &scale {
            s.check(self.dev, what, heads * dim)?;
            if s.overlaps(&x) {
                return refuse(what, "the scale overlaps the vectors".into());
            }
        }
        let (mut px, mut ps) = (x.address(), scale.map_or(0, |s| s.address()));
        let (mut v, mut h, mut d, mut c) = (vectors as u32, heads as u32, dim as u32, clip);
        let (mut pos, mut floor, mut alpha) = (late.pos as u64, late.floor, late.alpha);
        let mut args = [
            ptr(&mut px),
            ptr(&mut ps),
            num(&mut v),
            num(&mut h),
            num(&mut d),
            real(&mut c),
            big(&mut pos),
            real(&mut floor),
            real(&mut alpha),
        ];
        // SAFETY: the arguments match head_scale_clip(float*, const float*, unsigned,
        // unsigned, unsigned, float, unsigned long long, float, float). The kernel reads
        // and writes vectors * dim floats of `x` and reads heads * dim of `scale` unless it
        // is null; both were checked. One block per vector, of 128 threads: within the
        // kernel's shared array of 256.
        unsafe { self.scale_clip.launch((vectors as u32, 1, 1), (128, 1, 1), &mut args) }
    }

    /// For each of `heads` vectors of `dim` in `x`: `out = x / sqrt(mean(x^2) + eps)`.
    /// For a batch, `heads` counts the heads of every position.
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
        // One block per head, of 128 threads: within the kernel's shared array of 256.
        unsafe { self.rmsnorm_plain.launch((heads as u32, 1, 1), (128, 1, 1), &mut args) }
    }

    /// The short convolution with fixed taps (`channels * n` of them) of a batch of `batch`
    /// consecutive tokens, the first at `pos`, written to `out`. `current` and `out` hold
    /// the tokens one after another. `ring` holds the `n - 1` tokens before the batch.
    ///
    /// With a batch of one the token then takes its place in the ring. With a larger batch
    /// the ring is left as it was: call `ring_store` after this. Batches must be given in
    /// order of position.
    #[allow(clippy::too_many_arguments)]
    pub fn sconv_fixed(
        &self,
        current: Floats,
        ring: Floats,
        taps: Floats,
        out: Floats,
        n: usize,
        pos: usize,
        batch: usize,
    ) -> Result<(), CudaError> {
        let what = "sconv_fixed";
        let channels = self.conv_checks(what, current, ring, out, n, batch)?;
        taps.check(self.dev, what, channels * n)?;
        if taps.overlaps(&out) {
            return refuse(what, "the taps overlap the output".into());
        }
        let (mut pc, mut pr, mut pt, mut po) = (current.address(), ring.address(), taps.address(), out.address());
        let (mut c, mut taps_n, mut p, mut b) = (channels as u32, n as u32, pos as u64, batch as u32);
        let mut norm = 1.0 / ((n + 1) as f32).sqrt();
        let mut args = [
            ptr(&mut pc),
            ptr(&mut pr),
            ptr(&mut pt),
            ptr(&mut po),
            num(&mut c),
            num(&mut taps_n),
            big(&mut p),
            real(&mut norm),
            num(&mut b),
        ];
        // SAFETY: the arguments match sconv_fixed(const float*, float*, const float*, float*,
        // unsigned, unsigned, unsigned long long, float, unsigned). It reads batch * channels
        // floats of `current`, channels * n of `taps` and (n - 1) * channels of `ring`,
        // writes batch * channels to `out`, and writes the ring only for a batch of one; all
        // checked, and `out` overlaps none of the inputs.
        unsafe { self.sconv_fixed.launch((blocks((batch * channels) as u64, 256)?, 1, 1), (256, 1, 1), &mut args) }
    }

    /// The short convolution whose taps are `fixed_taps` (`n * channels`) plus `token_taps`
    /// (for each token of the batch, `heads * n`, shared by the channels of a head). The
    /// ring as for `sconv_fixed`.
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
        batch: usize,
    ) -> Result<(), CudaError> {
        let what = "sconv_dynamic";
        let channels = self.conv_checks(what, current, ring, out, n, batch)?;
        if heads == 0 || !channels.is_multiple_of(heads) {
            return refuse(what, format!("{channels} channels in {heads} heads"));
        }
        fixed_taps.check(self.dev, what, channels * n)?;
        token_taps.check(self.dev, what, batch * heads * n)?;
        if token_taps.overlaps(&out) || token_taps.overlaps(&ring) || fixed_taps.overlaps(&out) {
            return refuse(what, "the taps overlap something written".into());
        }
        let (mut pc, mut pr, mut pf, mut pt, mut po) =
            (current.address(), ring.address(), fixed_taps.address(), token_taps.address(), out.address());
        let (mut c, mut taps_n, mut h, mut p, mut b) =
            (channels as u32, n as u32, heads as u32, pos as u64, batch as u32);
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
            num(&mut b),
        ];
        // SAFETY: the arguments match sconv_dynamic(const float*, float*, const float*,
        // const float*, float*, unsigned, unsigned, unsigned, unsigned long long, float,
        // unsigned). Extents as for sconv_fixed, plus batch * heads * n floats of
        // `token_taps`; all checked.
        unsafe { self.sconv_dynamic.launch((blocks((batch * channels) as u64, 256)?, 1, 1), (256, 1, 1), &mut args) }
    }

    /// Checks what both convolutions share and returns the number of channels.
    fn conv_checks(
        &self,
        what: &str,
        current: Floats,
        ring: Floats,
        out: Floats,
        n: usize,
        batch: usize,
    ) -> Result<usize, CudaError> {
        if n == 0 || n > 64 || batch == 0 || batch > MAX_BATCH || !current.len.is_multiple_of(batch) {
            return refuse(what, format!("{n} taps, {} floats for a batch of {batch}", current.len));
        }
        let channels = current.len / batch;
        if channels == 0 || channels > 1 << 24 {
            return refuse(what, format!("{channels} channels"));
        }
        current.check(self.dev, what, batch * channels)?;
        out.check(self.dev, what, batch * channels)?;
        if n > 1 {
            ring.check(self.dev, what, (n - 1) * channels)?;
            if ring.overlaps(&current) || ring.overlaps(&out) {
                return refuse(what, "the ring overlaps the tokens or the output".into());
            }
        }
        if out.overlaps(&current) {
            return refuse(what, "the output overlaps the tokens".into());
        }
        Ok(channels)
    }

    /// After a convolution over a batch larger than one: the batch's last `n - 1` tokens
    /// take their places in `ring`. After a batch of one, which stored its token itself,
    /// it stores the same token again.
    pub fn ring_store(
        &self,
        current: Floats,
        ring: Floats,
        n: usize,
        pos: usize,
        batch: usize,
    ) -> Result<(), CudaError> {
        let what = "ring_store";
        if n == 0 || n > 64 || batch == 0 || batch > MAX_BATCH || !current.len.is_multiple_of(batch) {
            return refuse(what, format!("{n} taps, {} floats for a batch of {batch}", current.len));
        }
        if n == 1 {
            return Ok(());
        }
        let channels = current.len / batch;
        if channels == 0 || channels > 1 << 24 {
            return refuse(what, format!("{channels} channels"));
        }
        current.check(self.dev, what, batch * channels)?;
        ring.check(self.dev, what, (n - 1) * channels)?;
        if ring.overlaps(&current) {
            return refuse(what, "the ring overlaps the tokens".into());
        }
        let (mut pc, mut pr) = (current.address(), ring.address());
        let (mut c, mut taps_n, mut p, mut b) = (channels as u32, n as u32, pos as u64, batch as u32);
        let mut args = [ptr(&mut pc), ptr(&mut pr), num(&mut c), num(&mut taps_n), big(&mut p), num(&mut b)];
        // SAFETY: the arguments match ring_store(const float*, float*, unsigned, unsigned,
        // unsigned long long, unsigned). It reads batch * channels floats of `current` and
        // writes at most (n - 1) * channels of `ring`, each slot from one token; all checked.
        unsafe { self.ring_store.launch((blocks((batch * channels) as u64, 256)?, 1, 1), (256, 1, 1), &mut args) }
    }

    /// The keys or values of a batch, `rows` (one row of `width` per token), go to their
    /// rows of `history` (`cap` rows): the token at position p to row `p % cap`.
    pub fn store_rows(
        &self,
        rows: Floats,
        history: Floats,
        width: usize,
        cap: usize,
        pos: usize,
        batch: usize,
    ) -> Result<(), CudaError> {
        let what = "store_rows";
        // More tokens than rows would write one row twice in one launch.
        if width == 0 || cap == 0 || batch == 0 || batch > MAX_BATCH || batch > cap || width > 1 << 24 {
            return refuse(what, format!("{batch} rows of {width} into a history of {cap}"));
        }
        rows.check(self.dev, what, batch * width)?;
        history.check(self.dev, what, cap * width)?;
        if rows.overlaps(&history) {
            return refuse(what, "the rows are inside the history".into());
        }
        let (mut pr, mut ph, mut w) = (rows.address(), history.address(), width as u32);
        let (mut c, mut p, mut b) = (cap as u64, pos as u64, batch as u32);
        let mut args = [ptr(&mut pr), ptr(&mut ph), num(&mut w), big(&mut c), big(&mut p), num(&mut b)];
        // SAFETY: the arguments match store_rows(const float*, float*, unsigned, unsigned
        // long long, unsigned long long, unsigned). It reads batch * width floats of `rows`
        // and writes that many into rows below `cap` of `history`, no row twice; all checked.
        unsafe { self.store_rows.launch((blocks((batch * width) as u64, 256)?, 1, 1), (256, 1, 1), &mut args) }
    }

    /// Attention for a batch of `batch` query tokens, the first at position `pos`, whose
    /// own keys and values are already in the history: for each query and head, `out` is
    /// the sum over the tokens it sees of weight times value, the weights being a softmax
    /// over those tokens' scores and the head's sink score.
    ///
    /// A query at position p sees all tokens up to p, or the last `window` of them.
    /// `q` holds `heads * dim` floats per query, `relative.rel` `heads * rel_dim`, `sinks`
    /// `heads`. `weights` is scratch of `heads * attend_scratch(pos, window, batch)`.
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
        window: Option<usize>,
        batch: usize,
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
        if batch == 0 || batch > MAX_BATCH || window == Some(0) {
            return refuse(what, format!("a batch of {batch} with a window of {window:?}"));
        }
        // Every token the first query sees must still be in its row after the batch's own
        // tokens were stored: a token's row is taken by the token `cap` positions later.
        let seen_first = window.map_or(pos + 1, |w| (pos + 1).min(w));
        if seen_first + batch - 1 > history.cap {
            return refuse(
                what,
                format!(
                    "position {pos} sees {seen_first} tokens and its batch has {batch}: more than a history of {}",
                    history.cap
                ),
            );
        }
        let stride = Self::attend_scratch(pos, window, batch) / batch;
        let (width, runs) = (kv_heads * dim, batch * heads);
        q.check(self.dev, what, runs * dim)?;
        out.check(self.dev, what, runs * dim)?;
        history.keys.check(self.dev, what, history.cap * width)?;
        history.values.check(self.dev, what, history.cap * width)?;
        rel.check(self.dev, what, runs * rel_dim)?;
        table.check(self.dev, what, rel_dim * reach)?;
        sinks.check(self.dev, what, runs)?;
        weights.check(self.dev, what, runs * stride)?;
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
        let (mut p, mut win, mut cap, mut st) =
            (pos as u64, window.map_or(0, |w| w as u64), history.cap as u64, stride as u64);
        let mut b = batch as u32;
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
            big(&mut win),
            big(&mut cap),
            big(&mut st),
            num(&mut b),
        ];
        // SAFETY (the three launches): each argument list matches its kernel's parameters in
        // order and type. attn_scores reads the queries, the keys' rows of the tokens each
        // query sees (rows below cap), the relative vectors, columns below `reach` of the
        // table and the sinks, and writes into each (query, head) run of `stride` floats at
        // most seen + 1 entries, where seen + 1 is at most `stride`. attn_softmax rewrites
        // those entries in place, one block of 256 threads per run. attn_values reads them
        // and the values' rows and writes runs * dim floats to `out`. Every extent was
        // checked above, and nothing written overlaps anything read.
        unsafe { self.scores.launch((blocks((runs * stride) as u64, 256)?, 1, 1), (256, 1, 1), &mut args)? };
        let mut args = [ptr(&mut pw), num(&mut h), big(&mut p), big(&mut win), big(&mut st), num(&mut b)];
        unsafe { self.softmax.launch((runs as u32, 1, 1), (256, 1, 1), &mut args)? };
        let mut args = [
            ptr(&mut pw),
            ptr(&mut pv),
            ptr(&mut po),
            num(&mut h),
            num(&mut kv),
            num(&mut d),
            big(&mut p),
            big(&mut win),
            big(&mut cap),
            big(&mut st),
            num(&mut b),
        ];
        unsafe { self.values.launch((blocks((runs * dim) as u64, 256)?, 1, 1), (256, 1, 1), &mut args) }
    }

    /// How many floats of scratch `attend` needs per head for a batch of `batch` queries
    /// starting at `pos`: for each query, one more than the last query's view.
    pub fn attend_scratch(pos: usize, window: Option<usize>, batch: usize) -> usize {
        let last = pos + batch;
        batch * (window.map_or(last, |w| last.min(w)) + 1)
    }

    /// Softmax in place over each of `runs` runs of `count` scores, one after another.
    pub fn softmax_runs(&self, weights: Floats, runs: usize, count: usize) -> Result<(), CudaError> {
        let what = "attn_softmax";
        if runs == 0 || count < 2 || runs > 1 << 16 {
            return refuse(what, format!("{runs} runs of {count}; a run has at least 2 scores"));
        }
        weights.check(self.dev, what, runs * count)?;
        // In the kernel's terms: one head, a window of count - 1 tokens that every position
        // has filled, and the sink as the last score of a run.
        let (mut pw, mut h, mut p, mut win) = (weights.address(), 1u32, count as u64, count as u64 - 1);
        let (mut st, mut b) = (count as u64, runs as u32);
        let mut args = [ptr(&mut pw), num(&mut h), big(&mut p), big(&mut win), big(&mut st), num(&mut b)];
        // SAFETY: the arguments match attn_softmax(float*, unsigned, unsigned long long,
        // unsigned long long, unsigned long long, unsigned). One block of 256 threads per run
        // reads and writes window + 1 = count floats of its run of `count`; runs * count
        // floats were checked.
        unsafe { self.softmax.launch((runs as u32, 1, 1), (256, 1, 1), &mut args) }
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
