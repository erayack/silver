use std::arch::x86_64::{
    __m256i, __m512i, __mmask8, _mm256_loadu_si256, _mm512_and_si512, _mm512_cvtepu32_epi64,
    _mm512_load_si512, _mm512_set1_epi64, _mm512_store_si512, _mm512_test_epi64_mask,
};

use crate::{
    bls::{
        constants::LIMBS,
        fp8::{Fp8, LANES},
    },
    kzg::msm8::{buckets::NEGATED, store::PointStore},
};

/// An `Fp8` in memory.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Fp8Memory([[u64; LANES]; LIMBS]);

impl Fp8Memory {
    const ZERO: Self = Self([[0; LANES]; LIMBS]);

    #[target_feature(enable = "avx512f,avx512ifma")]
    fn of(value: &Fp8) -> Self {
        let mut memory = Self::ZERO;
        for j in 0..LIMBS {
            // SAFETY: each limb row is 64 bytes and `Fp8Memory` is 64-aligned.
            unsafe { _mm512_store_si512(memory.0[j].as_mut_ptr().cast(), value.0[j]) };
        }
        memory
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    fn load(&self) -> Fp8 {
        let mut value = Fp8::zero();
        for j in 0..LIMBS {
            // SAFETY: as in `of`.
            value.0[j] = unsafe { _mm512_load_si512(self.0[j].as_ptr().cast()) };
        }
        value
    }
}

/// Pending affine additions `out = a + b`, where `NEGATED` on a reference to
/// `a` or `b` negates the point. One inversion serves all of them, by prefix
/// products over groups of eight.
#[derive(Default)]
pub struct BatchedAdds {
    a: Vec<u32>,
    b: Vec<u32>,
    out: Vec<u32>,
    dx: Vec<Fp8Memory>,
    prefix: Vec<Fp8Memory>,
    xa: Vec<Fp8Memory>,
    xb: Vec<Fp8Memory>,
}

impl BatchedAdds {
    pub fn clear(&mut self) {
        self.a.clear();
        self.b.clear();
        self.out.clear();
    }

    pub fn push(&mut self, a: u32, b: u32, out: u32) {
        self.a.push(a);
        self.b.push(b);
        self.out.push(out);
    }

    pub fn is_empty(&self) -> bool {
        self.a.is_empty()
    }

    /// Executes the additions. False when some pair has equal x coordinates,
    /// which the addition formula cannot take.
    #[target_feature(enable = "avx512f,avx512ifma")]
    pub fn run(&mut self, store: &mut PointStore) -> bool {
        let groups = self.a.len().div_ceil(LANES);
        let last = self.a.len() - 1;
        let padded = groups * LANES;
        self.a.resize(padded, self.a[last]);
        self.b.resize(padded, self.b[last]);
        self.out.resize(padded, self.out[last]);
        self.dx.resize(groups, Fp8Memory::ZERO);
        self.prefix.resize(groups, Fp8Memory::ZERO);
        self.xa.resize(groups, Fp8Memory::ZERO);
        self.xb.resize(groups, Fp8Memory::ZERO);

        let mut product = Fp8::one();
        for g in 0..groups {
            let (xa, _) = store.gather(self.indices(&self.a, g));
            let (xb, _) = store.gather(self.indices(&self.b, g));
            let dx = xb.sub(&xa);
            if dx.is_zero_mask() != 0 {
                return false;
            }
            product = product.mul(&dx);
            self.dx[g] = Fp8Memory::of(&dx);
            self.prefix[g] = Fp8Memory::of(&product);
            self.xa[g] = Fp8Memory::of(&xa);
            self.xb[g] = Fp8Memory::of(&xb);
        }

        let mut inverse = product.inverse();
        for g in (0..groups).rev() {
            let dx_inverse = if g == 0 { inverse } else { inverse.mul(&self.prefix[g - 1].load()) };
            inverse = inverse.mul(&self.dx[g].load());
            self.finish_group(g, &dx_inverse, store);
        }
        true
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    fn finish_group(&self, g: usize, dx_inverse: &Fp8, store: &mut PointStore) {
        let (a, b) = (self.indices(&self.a, g), self.indices(&self.b, g));
        let (xa, xb) = (self.xa[g].load(), self.xb[g].load());
        let (ya, yb) = (store.gather_y(a), store.gather_y(b));
        let ya = Fp8::select(self.negated(&self.a, g), &ya, &ya.neg());
        let yb = Fp8::select(self.negated(&self.b, g), &yb, &yb.neg());
        let slope = yb.sub(&ya).mul(dx_inverse);
        let x3 = slope.square().sub(&xa).sub(&xb);
        let y3 = slope.mul(&xa.sub(&x3)).sub(&ya);
        store.scatter(self.indices(&self.out, g), &x3, &y3);
    }

    /// The references of group `g`, without their sign.
    #[target_feature(enable = "avx512f,avx512ifma")]
    fn indices(&self, references: &[u32], g: usize) -> __m512i {
        let widened = widen(references, g);
        _mm512_and_si512(widened, _mm512_set1_epi64(i64::from(!NEGATED)))
    }

    /// The lanes of group `g` whose reference is negated.
    #[target_feature(enable = "avx512f,avx512ifma")]
    fn negated(&self, references: &[u32], g: usize) -> __mmask8 {
        let widened = widen(references, g);
        _mm512_test_epi64_mask(widened, _mm512_set1_epi64(i64::from(NEGATED)))
    }
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn widen(references: &[u32], g: usize) -> __m512i {
    let group = &references[g * LANES..(g + 1) * LANES];
    // SAFETY: an unaligned load of the 32 bytes of `group`.
    let narrow = unsafe { _mm256_loadu_si256(group.as_ptr().cast::<__m256i>()) };
    _mm512_cvtepu32_epi64(narrow)
}
