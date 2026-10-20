//! The Jacobian formulas of `G1x8` on the lazy `Fp8` kernels, for the
//! multi-scalar and fixed-base multiplications. Coordinates enter and leave
//! normalised and below 2p; intermediate values are kept raw within the
//! bounds `fp8_lazy` states, and each product's operands are commented with
//! their bounds. The proved decoders keep `g1x8.rs`.

use std::arch::x86_64::__mmask8;

use crate::bls::{
    fp8::Fp8,
    fp8_lazy::{EIGHT_P, FOUR_P, TWO_P},
    g1x8::G1x8,
};

/// The multiples of p the lazy differences are offset by.
pub struct Offsets {
    pub two_p: Fp8,
    pub four_p: Fp8,
    pub eight_p: Fp8,
}

impl Offsets {
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn new() -> Self {
        Self {
            two_p: Fp8::splat_limbs(&TWO_P),
            four_p: Fp8::splat_limbs(&FOUR_P),
            eight_p: Fp8::splat_limbs(&EIGHT_P),
        }
    }

    /// Raw `v` below 4p, normalised below 2p.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn below_2p_from_4p(&self, v: &Fp8) -> Fp8 {
        v.normalise().cond_sub(&self.two_p)
    }

    /// Raw `v` below 8p, normalised below 2p.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn below_2p_from_8p(&self, v: &Fp8) -> Fp8 {
        v.normalise().cond_sub(&self.four_p).cond_sub(&self.two_p)
    }

    /// `2p − v` for `v` in (0, 2p), normalised: the negation of a nonzero
    /// value. No curve point has y = 0, as the group order is odd.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn neg(&self, v: &Fp8) -> Fp8 {
        self.two_p.sub_raw(v, &Fp8::zero()).normalise()
    }
}

impl G1x8 {
    /// madd-2007-bl with Z2 = 1. Undecided when the points coincide or z = 0.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn add_affine_lazy(&self, qx: &Fp8, qy: &Fp8, c: &Offsets) -> (Self, __mmask8) {
        let z1z1 = self.z.square_lazy();
        let u2 = qx.mul_lazy(&z1z1);
        let s2 = qy.mul_lazy(&self.z).mul_lazy(&z1z1);
        let h = u2.sub_raw(&self.x, &c.two_p).normalise(); // < 4p
        let hh = h.square_lazy(); // 16p² < R·p
        let i = hh.add_raw(&hh).add_raw(&hh).add_raw(&hh).normalise(); // < 8p
        let j = h.mul_lazy(&i); // 4p · 8p
        let r = s2.sub_raw(&self.y, &c.two_p);
        let r = r.add_raw(&r).normalise(); // < 8p
        let v = self.x.mul_lazy(&i); // 2p · 8p
        let rr = r.square_lazy(); // 64p²
        let x3 = c.below_2p_from_8p(&rr.sub_raw(&j, &c.two_p).sub_raw(&v.add_raw(&v), &c.four_p));
        let vx = v.sub_raw(&x3, &c.two_p).normalise(); // < 4p
        let yj = self.y.mul_lazy(&j);
        let y3 = c.below_2p_from_8p(&r.mul_lazy(&vx).sub_raw(&yj.add_raw(&yj), &c.four_p)); // 8p · 4p
        let zh = self.z.add_raw(&h).normalise(); // < 6p
        let z3 =
            c.below_2p_from_8p(&zh.square_lazy().sub_raw(&z1z1, &c.two_p).sub_raw(&hh, &c.two_p));
        let undecided = h.is_zero_mod_p_below_4p() | self.z.is_zero_mod_p_below_4p();
        (Self { x: x3, y: y3, z: z3 }, undecided)
    }

    /// add-2007-bl. Undecided when the points coincide or either z = 0.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn add_lazy(&self, q: &Self, c: &Offsets) -> (Self, __mmask8) {
        let z1z1 = self.z.square_lazy();
        let z2z2 = q.z.square_lazy();
        let u1 = self.x.mul_lazy(&z2z2);
        let u2 = q.x.mul_lazy(&z1z1);
        let s1 = self.y.mul_lazy(&q.z).mul_lazy(&z2z2);
        let s2 = q.y.mul_lazy(&self.z).mul_lazy(&z1z1);
        let h = u2.sub_raw(&u1, &c.two_p).normalise(); // < 4p
        let i = h.add_raw(&h).normalise().square_lazy(); // 64p²
        let j = h.mul_lazy(&i); // 4p · 2p
        let r = s2.sub_raw(&s1, &c.two_p);
        let r = r.add_raw(&r).normalise(); // < 8p
        let v = u1.mul_lazy(&i);
        let rr = r.square_lazy(); // 64p²
        let x3 = c.below_2p_from_8p(&rr.sub_raw(&j, &c.two_p).sub_raw(&v.add_raw(&v), &c.four_p));
        let vx = v.sub_raw(&x3, &c.two_p).normalise(); // < 4p
        let sj = s1.mul_lazy(&j);
        let y3 = c.below_2p_from_8p(&r.mul_lazy(&vx).sub_raw(&sj.add_raw(&sj), &c.four_p)); // 8p · 4p
        let zz = self.z.add_raw(&q.z).normalise(); // < 4p
        let zz = zz.square_lazy().sub_raw(&z1z1, &c.two_p).sub_raw(&z2z2, &c.two_p).normalise(); // < 6p
        let z3 = c.below_2p_from_4p(&zz.mul_lazy(&h)); // 6p · 4p
        let undecided = h.is_zero_mod_p_below_4p() |
            self.z.is_zero_mod_p_below_4p() |
            q.z.is_zero_mod_p_below_4p();
        (Self { x: x3, y: y3, z: z3 }, undecided)
    }

    /// dbl-2009-l for a chain of doublings: takes and gives x, y below 8p and
    /// z below 4p, normalised, with no zero test; a lane with y = 0 or z = 0
    /// gets z = 0 for the rest of the chain, which `tighten`'s caller tests.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn double_loose(&self, c: &Offsets) -> Self {
        let a = self.x.square_lazy(); // 64p²
        let b = self.y.square_lazy(); // 64p²
        let cc = b.square_lazy();
        let xb = self.x.add_raw(&b).normalise(); // < 10p
        let d = c.below_2p_from_8p(&xb.square_lazy().sub_raw(&a, &c.two_p).sub_raw(&cc, &c.two_p)); // 100p²
        let d = c.below_2p_from_4p(&d.add_raw(&d));
        let e = a.add_raw(&a).add_raw(&a).normalise(); // < 6p
        let f = e.square_lazy(); // 36p²
        let x3 = f.sub_raw(&d.add_raw(&d), &c.four_p).normalise(); // (0, 6p)
        let dx = d.sub_raw(&x3, &c.eight_p).normalise(); // < 10p
        let c8 = c.below_2p_from_8p(&cc.add_raw(&cc).add_raw(&cc).add_raw(&cc));
        let y3 = e.mul_lazy(&dx).sub_raw(&c8.add_raw(&c8), &c.four_p).normalise(); // 6p · 10p; (0, 6p)
        let yz = self.y.mul_lazy(&self.z); // 8p · 4p
        let z3 = yz.add_raw(&yz).normalise(); // < 4p
        Self { x: x3, y: y3, z: z3 }
    }

    /// From `double_loose`'s ranges back below 2p.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub fn tighten(&self, c: &Offsets) -> Self {
        Self {
            x: self.x.cond_sub(&c.four_p).cond_sub(&c.two_p),
            y: self.y.cond_sub(&c.four_p).cond_sub(&c.two_p),
            z: self.z.cond_sub(&c.two_p),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::bls::{
        constants::{P, P_U64},
        fp8::{LANES, Limbs, pack64, unpack52},
        simd_available,
    };

    fn random_below_p(rng: &mut ChaCha8Rng) -> Limbs {
        loop {
            let mut words: [u64; 6] = rng.r#gen();
            words[5] &= (1 << 61) - 1;
            if words.iter().rev().lt(P_U64.iter().rev()) {
                return unpack52(&words);
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    fn plain(p: &G1x8) -> [[[u64; 6]; LANES]; 3] {
        [p.x, p.y, p.z].map(|v| v.to_plain().map(|limbs| pack64(&limbs)))
    }

    /// Random field elements make the formulas' arithmetic identical whether
    /// or not the lanes are curve points.
    #[test]
    fn lazy_formulas_match_the_proved_ones() {
        if !simd_available() {
            return;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x61);
        for _ in 0..8 {
            let v: [[Limbs; LANES]; 5] =
                std::array::from_fn(|_| std::array::from_fn(|_| random_below_p(&mut rng)));
            // SAFETY: checked above.
            unsafe {
                let f = v.map(|limbs| Fp8::from_plain(&limbs));
                let p = G1x8 { x: f[0], y: f[1], z: f[2] };
                let q = G1x8 { x: f[3], y: f[4], z: f[2].add(&f[0]) };
                let c = Offsets::new();
                let mut loose = G1x8 { x: p.x, y: p.y, z: p.z };
                let mut tight = G1x8 { x: p.x, y: p.y, z: p.z };
                for _ in 0..5 {
                    loose = loose.double_loose(&c);
                    tight = tight.double().0;
                }
                assert!(plain(&tight) == plain(&loose.tighten(&c)), "double_loose chain");
                let (want, wu) = p.add_affine(&f[3], &f[4]);
                let (got, gu) = p.add_affine_lazy(&f[3], &f[4], &c);
                assert!(plain(&want) == plain(&got) && wu == gu, "add_affine");
                let (want, wu) = p.add(&q);
                let (got, gu) = p.add_lazy(&q, &c);
                assert!(plain(&want) == plain(&got) && wu == gu, "add");
            }
        }
    }

    fn below(multiple_of_p: Limbs) -> Limbs {
        let mut limbs = multiple_of_p;
        limbs[0] -= 1;
        limbs
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    fn splat_value(limbs: Limbs) -> Fp8 {
        Fp8::load(&[limbs; LANES])
    }

    #[test]
    fn lazy_formulas_match_the_proved_ones_at_their_input_bounds() {
        if !simd_available() {
            return;
        }
        let p_minus_one = below(P);
        let (two_p, four_p, eight_p) = (below(TWO_P), below(FOUR_P), below(EIGHT_P));
        // SAFETY: checked above.
        unsafe {
            let c = Offsets::new();
            let proved = |x: Limbs, y: Limbs, z: Limbs| G1x8 {
                x: splat_value(x),
                y: splat_value(y),
                z: splat_value(z),
            };

            let p = proved(two_p, two_p, two_p);
            let reference = proved(p_minus_one, p_minus_one, p_minus_one);
            let (qx, qy) = (splat_value(two_p), splat_value(unpack52(&[5, 0, 0, 0, 0, 0])));
            let (qx_reference, qy_reference) = (splat_value(p_minus_one), qy);
            let (want, wu) = reference.add_affine(&qx_reference, &qy_reference);
            let (got, gu) = p.add_affine_lazy(&qx, &qy, &c);
            assert!(plain(&want) == plain(&got) && wu == gu, "add_affine");

            let q = proved(two_p, unpack52(&[7, 0, 0, 0, 0, 0]), two_p);
            let q_reference = proved(p_minus_one, unpack52(&[7, 0, 0, 0, 0, 0]), p_minus_one);
            let (want, wu) = reference.add(&q_reference);
            let (got, gu) = p.add_lazy(&q, &c);
            assert!(plain(&want) == plain(&got) && wu == gu, "add");

            let mut loose = proved(eight_p, eight_p, four_p);
            let mut tight = reference.double().0;
            loose = loose.double_loose(&c);
            assert!(plain(&tight) == plain(&loose.tighten(&c)), "double_loose");
            for _ in 0..4 {
                loose = loose.double_loose(&c);
                tight = tight.double().0;
            }
            assert!(plain(&tight) == plain(&loose.tighten(&c)), "double_loose chain");
        }
    }
}
