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
    use crate::{
        bls::simd_available,
        kzg::{
            BYTES_PER_BLOB, BYTES_PER_CELL, BYTES_PER_FIELD_ELEMENT, CELLS_PER_EXT_BLOB,
            FIELD_ELEMENTS_PER_BLOB, FIELD_ELEMENTS_PER_CELL,
            blst_ops::{fr_from_u64, fr_inverse, fr_mul},
            domain::{inverse_roots, roots_of_unity},
            fr8::{Fr8, Fr8Soa, LANES, LIMBS, R_U64, pack64, unpack52},
            fr8_fft::{
                BLOCK_VECTORS, FIRST_WIDE_STAGE, dif_last_stage_canonical, dif_stage,
                dit_first_stage, dit_stage, layout_t, soa_of, transpose_blocks, twiddles,
            },
        },
    };

    const LOG_BLOB: usize = 12;
    const VECTORS: usize = FIELD_ELEMENTS_PER_BLOB / LANES;

    pub struct SimdCellExtender {
        /// Inverse roots.
        dit: [Vec<Fr8Soa>; LOG_BLOB],
        dif: [Vec<Fr8Soa>; LOG_BLOB],
        /// ω^k / 4096 for ω of order 8192, k < 4096, in layout S.
        pointwise: Vec<Fr8Soa>,
        buf: Vec<Fr8Soa>,
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
}
