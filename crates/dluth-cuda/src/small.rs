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

/// Most candidates one `blend_many` takes.
pub const BLEND_MAX: usize = 16;
/// Most vectors one `normalize_many` takes.
pub const MANY: usize = 8;
/// Most queries one `blend_many` takes.
pub const BLEND_QUERIES: usize = 4;

/// The small-operation kernels of a loaded module.
pub struct Small<'m> {
    dev: &'m CudaDevice,
    add: Kernel<'m>,
    rmsnorm: Kernel<'m>,
    blend_scores: Kernel<'m>,
    gate_merge: Kernel<'m>,
    softcap: Kernel<'m>,
    normalize_many: Kernel<'m>,
    copy: Kernel<'m>,
    blend_many: Kernel<'m>,
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
            normalize_many: module.kernel("normalize_many")?,
            copy: module.kernel("vec_copy")?,
            blend_many: module.kernel("blend_many")?,
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

    /// `out = x` for `out.len` floats.
    pub fn copy(&self, x: Floats, out: Floats) -> Result<(), CudaError> {
        let n = out.len;
        self.runs("vec_copy", n, &[x], &out)?;
        let (mut px, mut po, mut count) = (x.address(), out.address(), n as u32);
        let mut args = [ptr(&mut px), ptr(&mut po), num(&mut count)];
        // SAFETY: the arguments match vec_copy(const float*, float*, unsigned); it reads n
        // floats of `x` and writes n to `out`, both checked.
        unsafe { self.copy.launch(grid(n), (256, 1, 1), &mut args) }
    }

    /// For each of the vectors of `weights.len` floats in `x`:
    /// `out[i] = x[i] / sqrt(mean(x^2) + eps) * weights[i]`.
    pub fn rmsnorm(&self, x: Floats, weights: Floats, out: Floats, eps: f32) -> Result<(), CudaError> {
        let what = "rmsnorm_weighted";
        let (n, total) = (weights.len, out.len);
        if n == 0 || !total.is_multiple_of(n) || total / n > 1 << 16 {
            return refuse(what, format!("{total} floats in vectors of {n}"));
        }
        self.runs(what, total, &[x], &out)?;
        weights.check(self.dev, what, n)?;
        if weights.overlaps(&out) || eps < 0.0 || !eps.is_finite() {
            return refuse(what, format!("eps {eps}, or the weights overlap the output"));
        }
        let (mut px, mut pw, mut po) = (x.address(), weights.address(), out.address());
        let (mut count, mut e, mut batch) = (n as u32, eps, (total / n) as u32);
        let mut args = [ptr(&mut px), ptr(&mut pw), ptr(&mut po), num(&mut count), real(&mut e), num(&mut batch)];
        // SAFETY: the arguments match rmsnorm_weighted(const float*, const float*, float*,
        // unsigned, float, unsigned). One block of 256 threads (within the kernel's shared
        // array) per vector reads n floats of `x` and of `weights` and writes n to `out`;
        // batch * n floats of `x` and `out` and n of `weights` were checked.
        unsafe { self.rmsnorm.launch((batch, 1, 1), (256, 1, 1), &mut args) }
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

    /// Up to `MANY` candidates, each a batch of `batch` vectors, normalized into `out` so
    /// that each position's candidates are one after another:
    /// `out[b][j] = x_j[b] / sqrt(mean(x_j[b]^2) + eps)`.
    pub fn normalize_many(&self, vectors: &[Floats], out: Floats, eps: f32, batch: usize) -> Result<(), CudaError> {
        let what = "normalize_many";
        let count = vectors.len();
        if count == 0
            || count > MANY
            || batch == 0
            || batch > 1 << 12
            || !out.len.is_multiple_of(count * batch)
            || eps < 0.0
            || !eps.is_finite()
        {
            return refuse(
                what,
                format!("{count} candidates of {batch} into {} floats, eps {eps}; at most {MANY}", out.len),
            );
        }
        let width = out.len / (count * batch);
        if width == 0 || width > 1 << 24 {
            return refuse(what, format!("vectors of {width}"));
        }
        out.check(self.dev, what, batch * count * width)?;
        let mut pointers = [0u64; MANY];
        for (slot, v) in pointers.iter_mut().zip(vectors) {
            v.check(self.dev, what, batch * width)?;
            if v.overlaps(&out) {
                return refuse(what, "the output overlaps an input".into());
            }
            *slot = v.address();
        }
        let (mut po, mut c, mut w, mut e, mut n) = (out.address(), count as u32, width as u32, eps, batch as u32);
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
            num(&mut n),
        ];
        // SAFETY: the arguments match normalize_many(const float* x8, float*, unsigned,
        // unsigned, float, unsigned). One block of 256 threads per vector reads `width`
        // floats, within batch * width of one of the first `count` pointers (the others are
        // never touched), and writes them into batch * count * width floats of `out`. All
        // checked.
        unsafe { self.normalize_many.launch(((count * batch) as u32, 1, 1), (256, 1, 1), &mut args) }
    }

    /// A whole blend for up to `BLEND_QUERIES` queries in one launch, from candidates that
    /// have not been normalized: for each position of the batch and each query, normalize
    /// the candidates, blend them as the query asks, and, where a norm is given for the
    /// query, normalize the blend with those weights.
    ///
    /// Each of the up to `BLEND_MAX` candidates holds `batch` vectors of `width`; each
    /// query and each norm is one vector of `width`; each output receives `batch` vectors.
    #[allow(clippy::too_many_arguments)]
    pub fn blend_many(
        &self,
        candidates: &[Floats],
        queries: &[Floats],
        norms: &[Option<Floats>],
        outs: &[Floats],
        width: usize,
        batch: usize,
        blend_eps: f32,
        norm_eps: f32,
    ) -> Result<(), CudaError> {
        let what = "blend_many";
        let (count, asked) = (candidates.len(), queries.len());
        if count == 0 || count > BLEND_MAX || asked == 0 || asked > BLEND_QUERIES {
            return refuse(
                what,
                format!("{count} candidates for {asked} queries; at most {BLEND_MAX} and {BLEND_QUERIES}"),
            );
        }
        if norms.len() != asked || outs.len() != asked {
            return refuse(what, format!("{asked} queries with {} norms and {} outputs", norms.len(), outs.len()));
        }
        if width == 0 || width > 1 << 24 || batch == 0 || batch > 1 << 12 {
            return refuse(what, format!("{batch} positions of {width}"));
        }
        let finite = |e: f32| e >= 0.0 && e.is_finite();
        if !finite(blend_eps) || !finite(norm_eps) {
            return refuse(what, format!("eps of {blend_eps} and {norm_eps}"));
        }
        let mut reads: Vec<&Floats> = Vec::new();
        for candidate in candidates {
            if candidate.len != batch * width {
                return refuse(what, format!("a candidate of {} floats for {batch} of {width}", candidate.len));
            }
            candidate.check(self.dev, what, batch * width)?;
            reads.push(candidate);
        }
        for vector in queries.iter().chain(norms.iter().flatten()) {
            if vector.len != width {
                return refuse(what, format!("a query or norm of {} floats for a width of {width}", vector.len));
            }
            vector.check(self.dev, what, width)?;
            reads.push(vector);
        }
        for (i, out) in outs.iter().enumerate() {
            if out.len != batch * width {
                return refuse(what, format!("an output of {} floats for {batch} of {width}", out.len));
            }
            out.check(self.dev, what, batch * width)?;
            if reads.iter().any(|read| read.overlaps(out)) || outs[..i].iter().any(|other| other.overlaps(out)) {
                return refuse(what, "an output overlaps an input or another output".into());
            }
        }
        // Unused slots are null: the kernel only touches the first `count` candidates and
        // the first `asked` of the rest, and a null norm means no normalization.
        let mut c = [0u64; BLEND_MAX];
        let (mut q, mut w, mut o) = ([0u64; BLEND_QUERIES], [0u64; BLEND_QUERIES], [0u64; BLEND_QUERIES]);
        c.iter_mut().zip(candidates).for_each(|(slot, f)| *slot = f.address());
        q.iter_mut().zip(queries).for_each(|(slot, f)| *slot = f.address());
        w.iter_mut().zip(norms).for_each(|(slot, f)| *slot = f.map_or(0, |f| f.address()));
        o.iter_mut().zip(outs).for_each(|(slot, f)| *slot = f.address());
        let (mut n, mut wd, mut scale) = (count as u32, width as u32, 1.0 / (width as f32).sqrt());
        let (mut be, mut ne, mut b, mut nq) = (blend_eps, norm_eps, batch as u32, asked as u32);
        let mut args: Vec<*mut c_void> = Vec::with_capacity(35);
        args.extend(c.iter_mut().chain(&mut q).chain(&mut w).chain(&mut o).map(ptr));
        args.extend([
            num(&mut n),
            num(&mut wd),
            real(&mut scale),
            real(&mut be),
            real(&mut ne),
            num(&mut b),
            num(&mut nq),
        ]);
        // SAFETY: the arguments match blend_many(16 candidate pointers, 4 query pointers,
        // 4 norm pointers, 4 output pointers, unsigned, unsigned, float, float, float,
        // unsigned, unsigned), 35 in all. One block of 256 threads (8 whole warps, the
        // size of the kernel's shared rows) per (position, query) reads batch * width floats of each of
        // the first `count` candidates (at most BLEND_MAX, the number of shared rows),
        // `width` floats of its query and of its norm unless that is null, and writes
        // batch * width floats to its output. All were checked; no output overlaps
        // anything read or another output.
        unsafe { self.blend_many.launch(((batch * asked) as u32, 1, 1), (256, 1, 1), &mut args) }
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
