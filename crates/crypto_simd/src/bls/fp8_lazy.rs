//! Lazy-reduction kernels of `Fp8` for the multi-scalar multiplication and
//! the fixed-base multiplication: raw (uncarried) limbs, early-carry
//! Montgomery reduction, and squaring. The proved decoders keep the kernels
//! of `fp8.rs`.
//!
//! Ranges. A normalised operand has limbs below 2^52, so its value is below
//! 2^416 = R. The Montgomery product (a·b + m·p)/R is below 2p whenever
//! a·b < R·p: one operand below 2p and the other below 2^34·p, or both below
//! 2^17·p. Raw limbs are sums of at most 16 IFMA halves plus a carry, below
//! 2^57; raw sums and differences stay far below 2^63, where the signed carry
//! propagation of `normalise` is exact.

use std::arch::x86_64::{
    __m512i, __mmask8, _mm512_add_epi64, _mm512_and_si512, _mm512_cmpeq_epi64_mask,
    _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_mask_blend_epi64, _mm512_srai_epi64,
    _mm512_srli_epi64, _mm512_sub_epi64,
};

use crate::bls::{
    constants::{LIMBS, MASK52, P, P_INV52},
    fp8::{Fp8, Limbs, muladd_row, splat, splat_limbs, zero},
};

pub type Accumulator = [__m512i; 2 * LIMBS + 1];

/// `k·p` as normalised limbs, for k up to 2^12.
const fn times_p(k: u64) -> Limbs {
    let mut out = [0u64; LIMBS];
    let mut carry = 0u64;
    let mut j = 0;
    while j < LIMBS {
        let limb = P[j] * k + carry;
        out[j] = limb & MASK52;
        carry = limb >> 52;
        j += 1;
    }
    assert!(carry == 0);
    out
}

pub const TWO_P: Limbs = times_p(2);
pub const FOUR_P: Limbs = times_p(4);
pub const EIGHT_P: Limbs = times_p(8);

/// `(t + lo(q·p₀)) >> 52` without the product: the low limb clears to zero,
/// so the sum carries exactly when `t`'s low limb is nonzero.
#[target_feature(enable = "avx512f,avx512ifma")]
#[inline]
fn carry_of_cleared(t: __m512i) -> __m512i {
    _mm512_srli_epi64::<52>(_mm512_add_epi64(t, splat(MASK52)))
}

impl Fp8 {
    /// The 16-limb product, unreduced.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn product(&self, rhs: &Self) -> Accumulator {
        let mut t = [zero(); 2 * LIMBS + 1];
        for i in 0..LIMBS {
            muladd_row(&mut t, &self.0, rhs.0[i], i);
        }
        t
    }

    /// The 16-limb square: the off-diagonal products once, doubled, then the
    /// diagonal. 72 IFMA instead of 128.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn square_product(&self) -> Accumulator {
        let a = &self.0;
        let mut t = [zero(); 2 * LIMBS + 1];
        for i in 0..LIMBS {
            for j in i + 1..LIMBS {
                t[i + j] = _mm512_madd52lo_epu64(t[i + j], a[i], a[j]);
                t[i + j + 1] = _mm512_madd52hi_epu64(t[i + j + 1], a[i], a[j]);
            }
        }
        for k in 1..2 * LIMBS {
            t[k] = _mm512_add_epi64(t[k], t[k]);
        }
        for i in 0..LIMBS {
            t[2 * i] = _mm512_madd52lo_epu64(t[2 * i], a[i], a[i]);
            t[2 * i + 1] = _mm512_madd52hi_epu64(t[2 * i + 1], a[i], a[i]);
        }
        t
    }

    /// Montgomery-reduces `t` to raw limbs.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn reduce(mut t: Accumulator) -> Self {
        let p = splat_limbs(&P);
        let pinv = splat(P_INV52);
        for i in 0..LIMBS {
            let carry = carry_of_cleared(t[i]);
            let q = _mm512_madd52lo_epu64(zero(), t[i], pinv);
            t[i + 1] = _mm512_add_epi64(t[i + 1], carry);
            for j in 1..LIMBS {
                t[i + j] = _mm512_madd52lo_epu64(t[i + j], p[j], q);
            }
            for j in 0..LIMBS {
                t[i + j + 1] = _mm512_madd52hi_epu64(t[i + j + 1], p[j], q);
            }
        }
        Self(std::array::from_fn(|j| t[LIMBS + j]))
    }

    /// Montgomery product with raw output limbs; see the module ranges.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn mul_raw(&self, rhs: &Self) -> Self {
        Self::reduce(self.product(rhs))
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn mul_lazy(&self, rhs: &Self) -> Self {
        self.mul_raw(rhs).normalise()
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn square_raw(&self) -> Self {
        Self::reduce(self.square_product())
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn square_lazy(&self) -> Self {
        self.square_raw().normalise()
    }

    /// Limb-wise sum, no carries.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn add_raw(&self, rhs: &Self) -> Self {
        Self(std::array::from_fn(|j| _mm512_add_epi64(self.0[j], rhs.0[j])))
    }

    /// Limb-wise `self + c − rhs`, no carries; limbs may go negative. `c` is
    /// a multiple of p at or above `rhs`'s bound.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn sub_raw(&self, rhs: &Self, c: &Self) -> Self {
        Self(std::array::from_fn(|j| {
            _mm512_sub_epi64(_mm512_add_epi64(self.0[j], c.0[j]), rhs.0[j])
        }))
    }

    /// Signed carry propagation into 52-bit limbs, for a value in [0, 2^416).
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn normalise(&self) -> Self {
        let mask = splat(MASK52);
        let mut s = self.0;
        for j in 0..LIMBS - 1 {
            let carry = _mm512_srai_epi64::<52>(s[j]);
            s[j] = _mm512_and_si512(s[j], mask);
            s[j + 1] = _mm512_add_epi64(s[j + 1], carry);
        }
        Self(s)
    }

    /// `self − c` normalised, and the lanes where it borrowed.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    fn sub_borrow(&self, c: &Self) -> (Self, __mmask8) {
        let mask = splat(MASK52);
        let mut d = [zero(); LIMBS];
        let mut borrow = zero();
        for j in 0..LIMBS {
            let s = _mm512_add_epi64(_mm512_sub_epi64(self.0[j], c.0[j]), borrow);
            borrow = _mm512_srai_epi64::<63>(s);
            d[j] = _mm512_and_si512(s, mask);
        }
        (Self(d), _mm512_cmpeq_epi64_mask(borrow, splat(u64::MAX)))
    }

    /// For normalised `self` below 2c: `self − c` where `self ≥ c`.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn cond_sub(&self, c: &Self) -> Self {
        let (d, borrowed) = self.sub_borrow(c);
        Self(std::array::from_fn(|j| _mm512_mask_blend_epi64(borrowed, d.0[j], self.0[j])))
    }

    /// Lanes where normalised `self`, below 4p, is 0 mod p.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn is_zero_mod_p_below_4p(&self) -> __mmask8 {
        let canonical = self.cond_sub(&Self::splat_limbs(&TWO_P)).cond_sub(&Self::splat_limbs(&P));
        let mut k: __mmask8 = 0xff;
        for j in 0..LIMBS {
            k &= _mm512_cmpeq_epi64_mask(canonical.0[j], zero());
        }
        k
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::bls::{
        constants::P_U64,
        fp8::{LANES, pack64, unpack52},
        simd_available,
    };

    fn random_below(rng: &mut ChaCha8Rng, bound: &[u64; 6]) -> Limbs {
        loop {
            let mut words: [u64; 6] = rng.r#gen();
            words[5] &= (1 << 61) - 1;
            if words.iter().rev().lt(bound.iter().rev()) {
                return unpack52(&words);
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    fn plain(v: &Fp8) -> [[u64; 6]; LANES] {
        v.to_plain().map(|limbs| pack64(&limbs))
    }

    #[test]
    fn lazy_kernels_match_mul() {
        if !simd_available() {
            return;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x1a27);
        let mut two_p = [0u64; 6];
        let mut carry = 0u64;
        for j in 0..6 {
            let (d, c1) = P_U64[j].overflowing_add(P_U64[j]);
            let (s, c2) = d.overflowing_add(carry);
            two_p[j] = s;
            carry = u64::from(c1) + u64::from(c2);
        }
        for _ in 0..16 {
            let a: [Limbs; LANES] = std::array::from_fn(|_| random_below(&mut rng, &two_p));
            let b: [Limbs; LANES] = std::array::from_fn(|_| random_below(&mut rng, &two_p));
            // SAFETY: checked above.
            unsafe {
                let (a, b) = (Fp8::load(&a), Fp8::load(&b));
                assert!(plain(&a.mul_lazy(&b)) == plain(&a.mul(&b)));
                assert!(plain(&a.square_lazy()) == plain(&a.mul(&a)));
                let sum = a.add_raw(&b).normalise();
                assert!(plain(&sum.cond_sub(&Fp8::splat_limbs(&TWO_P))) == plain(&a.add(&b)));
                let diff = a.sub_raw(&b, &Fp8::splat_limbs(&TWO_P)).normalise();
                assert!(plain(&diff.cond_sub(&Fp8::splat_limbs(&TWO_P))) == plain(&a.sub(&b)));
                assert_eq!(
                    a.sub_raw(&a, &Fp8::splat_limbs(&TWO_P)).normalise().is_zero_mod_p_below_4p(),
                    0xff
                );
                assert_eq!(diff.is_zero_mod_p_below_4p(), 0);
            }
        }
    }
}
