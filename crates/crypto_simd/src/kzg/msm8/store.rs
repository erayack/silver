use std::arch::x86_64::{
    __m512i, _mm512_i64gather_epi64, _mm512_i64scatter_epi64, _mm512_slli_epi64,
    _mm512_storeu_si512,
};

use blst::{blst_fp, blst_fp_from_uint64};

use crate::bls::{
    blst_p1_affine,
    constants::LIMBS,
    fp8::{Fp8, LANES, Limbs, unpack52},
};

const POINT_WORDS: usize = 2 * LIMBS;
const POINT_WORDS_LOG: u32 = POINT_WORDS.trailing_zeros();
const _: () = assert!(POINT_WORDS.is_power_of_two());

/// Affine points in R' = 2^416 form, one per `POINT_WORDS` words: the eight x
/// limbs, then the eight y limbs. A gather of one limb across eight points
/// touches eight cache lines that the other limbs then reuse.
pub struct PointStore {
    words: Vec<u64>,
    len: usize,
    /// 2^448 mod p. Multiplying blst's Montgomery limbs (R = 2^384) by it in R'
    /// form gives R' form: blst's Montgomery form of 2^64.
    from_blst: Limbs,
}

impl PointStore {
    pub fn new() -> Self {
        let mut two_pow_64 = blst_fp::default();
        // SAFETY: valid pointers; blst reads six words and writes one value.
        unsafe { blst_fp_from_uint64(&mut two_pow_64, [0, 1, 0, 0, 0, 0].as_ptr()) };
        Self { words: Vec::new(), len: 0, from_blst: unpack52(&two_pow_64.l) }
    }

    pub fn capacity(&self) -> usize {
        self.words.len() / POINT_WORDS
    }

    /// Empties the store and makes room for `capacity` points, keeping what it
    /// has allocated.
    pub fn reset(&mut self, capacity: usize) {
        self.words.resize(self.words.len().max(capacity * POINT_WORDS), 0);
        self.len = 0;
    }

    /// The index of the next point, which the caller then writes by `scatter`.
    pub fn allocate(&mut self) -> u32 {
        assert!(self.len < self.capacity(), "the store holds every point a round can add");
        self.len += 1;
        (self.len - 1) as u32
    }

    /// Appends the points, converted to R' form.
    #[target_feature(enable = "avx512f,avx512ifma")]
    pub fn push_blst(&mut self, points: &[blst_p1_affine]) {
        assert!(self.len + points.len() <= self.capacity(), "the store holds the points");
        let from_blst = Fp8::splat_limbs(&self.from_blst);
        let mut xs = [[0u64; LIMBS]; LANES];
        let mut ys = [[0u64; LIMBS]; LANES];
        for chunk in points.chunks(LANES) {
            for (lane, point) in chunk.iter().enumerate() {
                xs[lane] = unpack52(&point.x.l);
                ys[lane] = unpack52(&point.y.l);
            }
            let x = Fp8::load(&xs).mul(&from_blst);
            let y = Fp8::load(&ys).mul(&from_blst);
            for j in 0..LIMBS {
                let (x_limbs, y_limbs) = (lanes(x.0[j]), lanes(y.0[j]));
                for lane in 0..chunk.len() {
                    let at = (self.len + lane) * POINT_WORDS;
                    self.words[at + j] = x_limbs[lane];
                    self.words[at + LIMBS + j] = y_limbs[lane];
                }
            }
            self.len += chunk.len();
        }
    }

    /// The points of the blst Montgomery limbs, in R' form.
    #[target_feature(enable = "avx512f,avx512ifma")]
    pub fn fp8_from_blst(&self, values: &[[u64; 6]; LANES]) -> Fp8 {
        let limbs: [Limbs; LANES] = std::array::from_fn(|lane| unpack52(&values[lane]));
        Fp8::load(&limbs).mul(&Fp8::splat_limbs(&self.from_blst))
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    pub fn gather(&self, indices: __m512i) -> (Fp8, Fp8) {
        (self.gather_coordinate(indices, 0), self.gather_coordinate(indices, LIMBS))
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    pub fn gather_y(&self, indices: __m512i) -> Fp8 {
        self.gather_coordinate(indices, LIMBS)
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    pub fn scatter(&mut self, indices: __m512i, x: &Fp8, y: &Fp8) {
        let offsets = _mm512_slli_epi64::<POINT_WORDS_LOG>(indices);
        for j in 0..LIMBS {
            // SAFETY: the indices name allocated points, so every offset stays inside
            // `words`.
            unsafe {
                _mm512_i64scatter_epi64::<8>(
                    self.words.as_mut_ptr().add(j).cast(),
                    offsets,
                    x.0[j],
                );
                _mm512_i64scatter_epi64::<8>(
                    self.words.as_mut_ptr().add(LIMBS + j).cast(),
                    offsets,
                    y.0[j],
                );
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    fn gather_coordinate(&self, indices: __m512i, first_word: usize) -> Fp8 {
        let offsets = _mm512_slli_epi64::<POINT_WORDS_LOG>(indices);
        let mut out = Fp8::zero();
        for j in 0..LIMBS {
            // SAFETY: the indices name allocated points, so every offset stays inside
            // `words`.
            unsafe {
                out.0[j] = _mm512_i64gather_epi64::<8>(
                    offsets,
                    self.words.as_ptr().add(first_word + j).cast(),
                );
            }
        }
        out
    }
}

#[target_feature(enable = "avx512f,avx512ifma")]
pub fn lanes(v: __m512i) -> [u64; LANES] {
    let mut out = [0u64; LANES];
    // SAFETY: an unaligned store of the 64 bytes of `out`.
    unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), v) };
    out
}
