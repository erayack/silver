//! `CellExtender` on `Fr8`: the same bytes from an iterative inverse FFT, the
//! shift-and-scale by ω^k/4096, and a forward FFT, eight elements per vector.
//!
//! The blob's elements are the evaluations in bit-reversed order, so the
//! decimation-in-time inverse FFT reads them as they are and the
//! decimation-in-frequency forward FFT leaves its output bit-reversed, as the
//! cells want it: no permutation. Stages of length 2 to 8 pair elements within
//! a block of eight vectors, so they run on a transposed layout.
//!
//! Layout S: element i at vector i/8, lane i%8. Layout T: element i at vector
//! (i/64)·8 + i%8, lane (i/8)%8. A block transpose switches between them.

#[cfg(target_arch = "x86_64")]
pub use x86::SimdCellExtender;

#[cfg(not(target_arch = "x86_64"))]
pub struct SimdCellExtender;

#[cfg(not(target_arch = "x86_64"))]
impl SimdCellExtender {
    pub fn new() -> Option<Self> {
        None
    }

    pub fn compute(
        &mut self,
        _blob: &[u8; crate::kzg::BYTES_PER_BLOB],
        _cells: &mut [[u8; crate::kzg::BYTES_PER_CELL]; crate::kzg::CELLS_PER_EXT_BLOB],
    ) -> bool {
        unreachable!("`new` is None without x86_64")
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use blst::blst_fr;

    use crate::{
        bls::simd_available,
        kzg::{
            BYTES_PER_BLOB, BYTES_PER_CELL, BYTES_PER_FIELD_ELEMENT, CELLS_PER_EXT_BLOB,
            FIELD_ELEMENTS_PER_BLOB, FIELD_ELEMENTS_PER_CELL,
            blst_ops::{fr_from_u64, fr_inverse, fr_mul},
            domain::{inverse_roots, roots_of_unity},
            fr8::{Fr8, Fr8Soa, LANES, LIMBS, R_U64, R52, TWO_R52, pack64, to_r260, unpack52},
        },
    };

    const LOG_BLOB: usize = 12;
    const VECTORS: usize = FIELD_ELEMENTS_PER_BLOB / LANES;
    const BLOCK_VECTORS: usize = 8;
    const BLOCK_ELEMENTS: usize = BLOCK_VECTORS * LANES;
    /// Stages shorter than this pair elements within a block.
    const FIRST_WIDE_STAGE: usize = 3;

    pub struct SimdCellExtender {
        /// Stage s has length 2^(s+1). Below `FIRST_WIDE_STAGE`: one splat of
        /// ω^j per butterfly j. Above: vector v holds ω^(8v+l) in lane l.
        /// Inverse roots.
        dit: [Vec<Fr8Soa>; LOG_BLOB],
        dif: [Vec<Fr8Soa>; LOG_BLOB],
        /// ω^k / 4096 for ω of order 8192, k < 4096, in layout S.
        pointwise: Vec<Fr8Soa>,
        buf: Vec<Fr8Soa>,
    }

    fn twiddles(roots: &[blst_fr]) -> [Vec<Fr8Soa>; LOG_BLOB] {
        std::array::from_fn(|stage| {
            let step = FIELD_ELEMENTS_PER_BLOB / (2 << stage);
            let splat = |j: usize| {
                let limbs = to_r260(&roots[j * step]);
                Fr8Soa(limbs.map(|limb| [limb; LANES]))
            };
            if stage < FIRST_WIDE_STAGE {
                return (0..1 << stage).map(splat).collect();
            }
            (0..1 << (stage - FIRST_WIDE_STAGE))
                .map(|v| soa_of(|lane| &roots[(LANES * v + lane) * step]))
                .collect()
        })
    }

    fn soa_of<'a>(lane_value: impl Fn(usize) -> &'a blst_fr) -> Fr8Soa {
        let mut soa = Fr8Soa([[0; LANES]; LIMBS]);
        for lane in 0..LANES {
            let limbs = to_r260(lane_value(lane));
            for j in 0..LIMBS {
                soa.0[j][lane] = limbs[j];
            }
        }
        soa
    }

    impl SimdCellExtender {
        /// None without AVX-512 IFMA.
        pub fn new() -> Option<Self> {
            if !simd_available() {
                return None;
            }
            let roots = roots_of_unity(LOG_BLOB as u32);
            let inverse_size = fr_inverse(&fr_from_u64(FIELD_ELEMENTS_PER_BLOB as u64));
            let shifts = roots_of_unity(LOG_BLOB as u32 + 1);
            let scaled: Vec<_> = shifts[..FIELD_ELEMENTS_PER_BLOB]
                .iter()
                .map(|shift| fr_mul(shift, &inverse_size))
                .collect();
            Some(Self {
                dit: twiddles(&inverse_roots(&roots)),
                dif: twiddles(&roots),
                pointwise: (0..VECTORS).map(|v| soa_of(|lane| &scaled[LANES * v + lane])).collect(),
                buf: vec![Fr8Soa([[0; LANES]; LIMBS]); VECTORS],
            })
        }

        /// False exactly when `CellExtender::compute` is: some element is not
        /// below r. `cells` is then unchanged.
        pub fn compute(
            &mut self,
            blob: &[u8; BYTES_PER_BLOB],
            cells: &mut [[u8; BYTES_PER_CELL]; CELLS_PER_EXT_BLOB],
        ) -> bool {
            if !self.decode_blob(blob) {
                return false;
            }
            for (k, cell) in cells[..CELLS_PER_EXT_BLOB / 2].iter_mut().enumerate() {
                cell.copy_from_slice(&blob[k * BYTES_PER_CELL..(k + 1) * BYTES_PER_CELL]);
            }
            // SAFETY: `new` checked avx512f and avx512ifma.
            unsafe { self.transform() };
            self.write_extension(cells);
            true
        }

        /// Into layout T, as limbs of the plain value: the first multiplication
        /// by a twiddle in R' form brings it into Montgomery form.
        fn decode_blob(&mut self, blob: &[u8; BYTES_PER_BLOB]) -> bool {
            for (i, element) in blob.chunks_exact(BYTES_PER_FIELD_ELEMENT).enumerate() {
                let value: [u64; 4] = std::array::from_fn(|w| {
                    let at = BYTES_PER_FIELD_ELEMENT - 8 * (w + 1);
                    u64::from_be_bytes(element[at..at + 8].try_into().unwrap())
                });
                if value.iter().rev().ge(R_U64.iter().rev()) {
                    return false;
                }
                let limbs = unpack52(&value);
                let (vector, lane) = layout_t(i);
                for j in 0..LIMBS {
                    self.buf[vector].0[j][lane] = limbs[j];
                }
            }
            true
        }

        fn write_extension(&self, cells: &mut [[u8; BYTES_PER_CELL]; CELLS_PER_EXT_BLOB]) {
            for i in 0..FIELD_ELEMENTS_PER_BLOB {
                let (vector, lane) = layout_t(i);
                let limbs: [u64; LIMBS] = std::array::from_fn(|j| self.buf[vector].0[j][lane]);
                let value = pack64(&limbs);
                let cell = &mut cells[CELLS_PER_EXT_BLOB / 2 + i / FIELD_ELEMENTS_PER_CELL];
                let at = i % FIELD_ELEMENTS_PER_CELL * BYTES_PER_FIELD_ELEMENT;
                for (w, word) in value.iter().enumerate() {
                    let end = at + BYTES_PER_FIELD_ELEMENT - 8 * w;
                    cell[end - 8..end].copy_from_slice(&word.to_be_bytes());
                }
            }
        }

        #[target_feature(enable = "avx512f,avx512ifma")]
        unsafe fn transform(&mut self) {
            let Self { dit, dif, pointwise, buf } = self;

            for block in buf.chunks_exact_mut(BLOCK_VECTORS) {
                dit_first_stage(block);
                for stage in 1..FIRST_WIDE_STAGE {
                    dit_stage(block, &dit[stage], 1 << stage);
                }
            }
            transpose_blocks(buf);
            for stage in FIRST_WIDE_STAGE..LOG_BLOB {
                dit_stage(buf, &dit[stage], 1 << (stage - FIRST_WIDE_STAGE));
            }

            for (vector, scale) in buf.iter_mut().zip(pointwise.iter()) {
                Fr8::load(vector).normalise().mul(&Fr8::load(scale)).store(vector);
            }

            for stage in (FIRST_WIDE_STAGE..LOG_BLOB).rev() {
                dif_stage(buf, &dif[stage], 1 << (stage - FIRST_WIDE_STAGE));
            }
            transpose_blocks(buf);
            for block in buf.chunks_exact_mut(BLOCK_VECTORS) {
                for stage in (1..FIRST_WIDE_STAGE).rev() {
                    dif_stage(block, &dif[stage], 1 << stage);
                }
                dif_last_stage_canonical(block);
            }
        }
    }

    /// Vector and lane of element `i` in layout T.
    fn layout_t(i: usize) -> (usize, usize) {
        let (block, within) = (i / BLOCK_ELEMENTS, i % BLOCK_ELEMENTS);
        (block * BLOCK_VECTORS + within % LANES, within / LANES)
    }

    /// With R' = 2^260 and r < 2^255, (a·b + q·r) / R' < 2r when b < r and
    /// a < R', so the product of any partially reduced `b` stays below 2r. Each
    /// DIT stage then adds at most 2r to the running bound, reaching 24r over
    /// the eleven multiplying stages, below the 32r that `mul` takes.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    fn dit_butterfly(a: &mut Fr8Soa, b: &mut Fr8Soa, twiddle: &Fr8Soa) {
        let two_r = Fr8::splat(&TWO_R52);
        let (x, y) = (Fr8::load(a), Fr8::load(b));
        let product = y.normalise().mul(&Fr8::load(twiddle));
        x.add_raw(&product).store(a);
        x.sub_raw(&product, &two_r).store(b);
    }

    /// The sum of two values below 2r needs one conditional subtraction of 2r
    /// to stay in `mul`'s range; the difference, below 4r, goes straight into
    /// the product.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    fn dif_butterfly(a: &mut Fr8Soa, b: &mut Fr8Soa, twiddle: &Fr8Soa) {
        let two_r = Fr8::splat(&TWO_R52);
        let (x, y) = (Fr8::load(a), Fr8::load(b));
        x.add_raw(&y).normalise().cond_sub(&two_r).store(a);
        x.sub_raw(&y, &two_r).normalise().mul(&Fr8::load(twiddle)).store(b);
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    fn dit_stage(buf: &mut [Fr8Soa], twiddles: &[Fr8Soa], half_vectors: usize) {
        for group in buf.chunks_exact_mut(2 * half_vectors) {
            let (low, high) = group.split_at_mut(half_vectors);
            for ((a, b), twiddle) in low.iter_mut().zip(high).zip(twiddles) {
                dit_butterfly(a, b, twiddle);
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    fn dif_stage(buf: &mut [Fr8Soa], twiddles: &[Fr8Soa], half_vectors: usize) {
        for group in buf.chunks_exact_mut(2 * half_vectors) {
            let (low, high) = group.split_at_mut(half_vectors);
            for ((a, b), twiddle) in low.iter_mut().zip(high).zip(twiddles) {
                dif_butterfly(a, b, twiddle);
            }
        }
    }

    /// Length 2 needs no multiplication: inputs below r give outputs below 2r.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    fn dit_first_stage(block: &mut [Fr8Soa]) {
        let r = Fr8::splat(&R52);
        for pair in block.chunks_exact_mut(2) {
            let (x, y) = (Fr8::load(&pair[0]), Fr8::load(&pair[1]));
            x.add_raw(&y).store(&mut pair[0]);
            x.sub_raw(&y, &r).store(&mut pair[1]);
        }
    }

    /// Length 2 of the forward FFT, reducing the block's outputs below r.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    fn dif_last_stage_canonical(block: &mut [Fr8Soa]) {
        let (r, two_r) = (Fr8::splat(&R52), Fr8::splat(&TWO_R52));
        let canonical = |v: Fr8| v.normalise().cond_sub(&two_r).cond_sub(&r);
        for pair in block.chunks_exact_mut(2) {
            let (x, y) = (Fr8::load(&pair[0]), Fr8::load(&pair[1]));
            canonical(x.add_raw(&y)).store(&mut pair[0]);
            canonical(x.sub_raw(&y, &two_r)).store(&mut pair[1]);
        }
    }

    /// Layout S to T and back.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    fn transpose_blocks(buf: &mut [Fr8Soa]) {
        for block in buf.chunks_exact_mut(BLOCK_VECTORS) {
            Fr8::transpose_block(block);
        }
    }
}
