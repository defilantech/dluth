//! Small operations on whole vectors, on the device. Each starts its kernel and returns
//! without waiting, after checking that everything the kernel touches is inside its buffers.

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::floats::Floats;
use crate::module::{Kernel, Module};
use std::ffi::c_void;

fn refuse<T>(what: &str, why: String) -> Result<T, CudaError> {
    Err(CudaError::Refused(format!("{what}: {why}")))
}

fn ptr(p: &mut u64) -> *mut c_void {
    p as *mut u64 as *mut c_void
}
fn num(p: &mut u32) -> *mut c_void {
    p as *mut u32 as *mut c_void
}
fn real(p: &mut f32) -> *mut c_void {
    p as *mut f32 as *mut c_void
}

fn grid(n: usize) -> (u32, u32, u32) {
    ((n as u32).div_ceil(256), 1, 1)
}

/// Most candidates one `blend_mix` takes.
pub const BLEND_MAX: usize = 16;
/// Most vectors one `normalize_many` takes.
pub const MANY: usize = 8;

/// The small-operation kernels of a loaded module.
pub struct Small<'m> {
    dev: &'m CudaDevice,
    add: Kernel<'m>,
    rmsnorm: Kernel<'m>,
    blend_scores: Kernel<'m>,
    gate_merge: Kernel<'m>,
    softcap: Kernel<'m>,
    blend_mix: Kernel<'m>,
    normalize_many: Kernel<'m>,
}

impl<'m> Small<'m> {
    pub fn new(dev: &'m CudaDevice, module: &'m Module<'m>) -> Result<Self, CudaError> {
        Ok(Small {
            dev,
            add: module.kernel("vec_add")?,
            rmsnorm: module.kernel("rmsnorm_weighted")?,
            blend_scores: module.kernel("blend_scores")?,
            gate_merge: module.kernel("gate_merge")?,
            softcap: module.kernel("softcap_inplace")?,
            blend_mix: module.kernel("blend_mix")?,
            normalize_many: module.kernel("normalize_many")?,
        })
    }

    /// Checks `n` and that every run is `n` floats inside its buffer, and that `out`
    /// overlaps none of `reads`.
    fn runs(&self, what: &str, n: usize, reads: &[Floats], out: &Floats) -> Result<(), CudaError> {
        if n == 0 || n > u32::MAX as usize / 2 {
            return refuse(what, format!("{n} values"));
        }
        out.check(self.dev, what, n)?;
        for read in reads {
            read.check(self.dev, what, n)?;
            if read.overlaps(out) {
                return refuse(what, "the output overlaps an input".into());
            }
        }
        Ok(())
    }

    /// `out = a + b`.
    pub fn add(&self, a: Floats, b: Floats, out: Floats) -> Result<(), CudaError> {
        let n = out.len;
        self.runs("vec_add", n, &[a, b], &out)?;
        let (mut pa, mut pb, mut po, mut count) = (a.address(), b.address(), out.address(), n as u32);
        let mut args = [ptr(&mut pa), ptr(&mut pb), ptr(&mut po), num(&mut count)];
        // SAFETY: the arguments match vec_add(const float*, const float*, float*, unsigned);
        // it reads n floats of each input and writes n to `out`, all checked.
        unsafe { self.add.launch(grid(n), (256, 1, 1), &mut args) }
    }

    /// `out[i] = x[i] / sqrt(mean(x^2) + eps) * weights[i]`.
    pub fn rmsnorm(&self, x: Floats, weights: Floats, out: Floats, eps: f32) -> Result<(), CudaError> {
        let what = "rmsnorm_weighted";
        let n = out.len;
        self.runs(what, n, &[x, weights], &out)?;
        if eps < 0.0 || !eps.is_finite() {
            return refuse(what, format!("eps {eps}"));
        }
        let (mut px, mut pw, mut po, mut count, mut e) = (x.address(), weights.address(), out.address(), n as u32, eps);
        let mut args = [ptr(&mut px), ptr(&mut pw), ptr(&mut po), num(&mut count), real(&mut e)];
        // SAFETY: the arguments match rmsnorm_weighted(const float*, const float*, float*,
        // unsigned, float). One block of 256 threads (within the kernel's shared array)
        // reads n floats of `x` and `weights` and writes n to `out`, all checked.
        unsafe { self.rmsnorm.launch((1, 1, 1), (256, 1, 1), &mut args) }
    }

    /// `scores[j] = (query . candidate_j) / sqrt(width)` for `count` candidates of `width`
    /// floats, one after another in `candidates`.
    pub fn blend_scores(
        &self,
        query: Floats,
        candidates: Floats,
        scores: Floats,
        count: usize,
        width: usize,
    ) -> Result<(), CudaError> {
        let what = "blend_scores";
        if count == 0 || width == 0 || count > 1 << 16 || width > 1 << 24 {
            return refuse(what, format!("{count} candidates of {width}"));
        }
        query.check(self.dev, what, width)?;
        candidates.check(self.dev, what, count * width)?;
        scores.check(self.dev, what, count)?;
        if scores.overlaps(&query) || scores.overlaps(&candidates) {
            return refuse(what, "the scores overlap an input".into());
        }
        let (mut pq, mut pc, mut ps) = (query.address(), candidates.address(), scores.address());
        let (mut c, mut w, mut scale) = (count as u32, width as u32, 1.0 / (width as f32).sqrt());
        let mut args = [ptr(&mut pq), ptr(&mut pc), ptr(&mut ps), num(&mut c), num(&mut w), real(&mut scale)];
        // SAFETY: the arguments match blend_scores(const float*, const float*, float*,
        // unsigned, unsigned, float). One block of 256 threads per candidate reads `width`
        // floats of the query and of its candidate and writes one score; all checked.
        unsafe { self.blend_scores.launch((count as u32, 1, 1), (256, 1, 1), &mut args) }
    }

    /// A whole blend in one launch: `out = sum_j softmax_j((query . candidate_j) / sqrt(width))
    /// * candidate_j` over `count` candidates of `width` floats in `candidates`.
    pub fn blend_mix(
        &self,
        query: Floats,
        candidates: Floats,
        out: Floats,
        count: usize,
        width: usize,
    ) -> Result<(), CudaError> {
        let what = "blend_mix";
        if count == 0 || count > BLEND_MAX || width == 0 || width > 1 << 24 {
            return refuse(what, format!("{count} candidates of {width}; at most {BLEND_MAX} candidates"));
        }
        query.check(self.dev, what, width)?;
        candidates.check(self.dev, what, count * width)?;
        out.check(self.dev, what, width)?;
        if out.overlaps(&query) || out.overlaps(&candidates) {
            return refuse(what, "the output overlaps an input".into());
        }
        let (mut pq, mut pc, mut po) = (query.address(), candidates.address(), out.address());
        let (mut c, mut w, mut scale) = (count as u32, width as u32, 1.0 / (width as f32).sqrt());
        let mut args = [ptr(&mut pq), ptr(&mut pc), ptr(&mut po), num(&mut c), num(&mut w), real(&mut scale)];
        // SAFETY: the arguments match blend_mix(const float*, const float*, float*, unsigned,
        // unsigned, float). One block of 256 threads, the size of the kernel's shared rows,
        // with at most BLEND_MAX candidates, the number of those rows; it reads `width`
        // floats of the query and count * width of the candidates and writes `width` to
        // `out`. All checked.
        unsafe { self.blend_mix.launch((1, 1, 1), (256, 1, 1), &mut args) }
    }

    /// Up to `MANY` vectors of `out.len / vectors.len()` floats each, normalized one after
    /// another into `out`: `out[j] = x_j / sqrt(mean(x_j^2) + eps)`.
    pub fn normalize_many(&self, vectors: &[Floats], out: Floats, eps: f32) -> Result<(), CudaError> {
        let what = "normalize_many";
        let count = vectors.len();
        if count == 0 || count > MANY || !out.len.is_multiple_of(count) || eps < 0.0 || !eps.is_finite() {
            return refuse(what, format!("{count} vectors into {} floats, eps {eps}; at most {MANY} vectors", out.len));
        }
        let width = out.len / count;
        if width == 0 || width > 1 << 24 {
            return refuse(what, format!("vectors of {width}"));
        }
        out.check(self.dev, what, count * width)?;
        let mut pointers = [0u64; MANY];
        for (slot, v) in pointers.iter_mut().zip(vectors) {
            v.check(self.dev, what, width)?;
            if v.overlaps(&out) {
                return refuse(what, "the output overlaps an input".into());
            }
            *slot = v.address();
        }
        let (mut po, mut c, mut w, mut e) = (out.address(), count as u32, width as u32, eps);
        let [a, b, cc, d, f, g, h, i] = &mut pointers;
        let mut args = [
            ptr(a),
            ptr(b),
            ptr(cc),
            ptr(d),
            ptr(f),
            ptr(g),
            ptr(h),
            ptr(i),
            ptr(&mut po),
            num(&mut c),
            num(&mut w),
            real(&mut e),
        ];
        // SAFETY: the arguments match normalize_many(const float* x8, float*, unsigned,
        // unsigned, float). One block of 256 threads per vector reads `width` floats of the
        // first `count` pointers (the others are never touched) and writes count * width
        // floats to `out`. All checked.
        unsafe { self.normalize_many.launch((count as u32, 1, 1), (256, 1, 1), &mut args) }
    }

    /// `out[i] = (1 - tanh(forget[i])) * x[i] + tanh(gain[i]) * message[i]`.
    pub fn gate_merge(
        &self,
        x: Floats,
        message: Floats,
        forget: Floats,
        gain: Floats,
        out: Floats,
    ) -> Result<(), CudaError> {
        let n = out.len;
        self.runs("gate_merge", n, &[x, message, forget, gain], &out)?;
        let (mut px, mut pm, mut pf, mut pg, mut po) =
            (x.address(), message.address(), forget.address(), gain.address(), out.address());
        let mut count = n as u32;
        let mut args = [ptr(&mut px), ptr(&mut pm), ptr(&mut pf), ptr(&mut pg), ptr(&mut po), num(&mut count)];
        // SAFETY: the arguments match gate_merge(const float* x4, float*, unsigned); it
        // reads n floats of each input and writes n to `out`, all checked.
        unsafe { self.gate_merge.launch(grid(n), (256, 1, 1), &mut args) }
    }

    /// In place: `x[i] = cap * tanh(x[i] / cap)`.
    pub fn softcap(&self, x: Floats, cap: f32) -> Result<(), CudaError> {
        let what = "softcap_inplace";
        let n = x.len;
        if n > u32::MAX as usize / 2 || cap <= 0.0 || !cap.is_finite() {
            return refuse(what, format!("{n} values with a cap of {cap}"));
        }
        x.check(self.dev, what, n)?;
        let (mut px, mut count, mut c) = (x.address(), n as u32, cap);
        let mut args = [ptr(&mut px), num(&mut count), real(&mut c)];
        // SAFETY: the arguments match softcap_inplace(float*, unsigned, float); it reads and
        // writes n floats of `x`, checked.
        unsafe { self.softcap.launch(grid(n), (256, 1, 1), &mut args) }
    }
}
