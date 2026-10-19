use std::arch::x86_64::{__m512i, __mmask8, _mm512_loadu_si512};

use blst::{
    blst_fp, blst_p1, blst_p1_add_or_double, blst_p1_add_or_double_affine, blst_p1_affine,
    blst_p1_double, blst_p1_is_inf,
};

use crate::{
    bls::{
        fp8::{Fp8, LANES},
        g1x8::G1x8,
    },
    kzg::msm8::{
        buckets::{Buckets, NEGATED},
        store::PointStore,
    },
};

/// The sum S_j = Σ_b b·B_(j,b) of each window's buckets.
#[derive(Default)]
pub struct WindowSums {
    sums: Vec<blst_p1>,
}

impl WindowSums {
    /// Eight windows at a time, one per lane, with running sums T = Σ_(b' ≥ b)
    /// B_b' and S += T for b from the top bucket down to 1. False when a lane
    /// sums to infinity in blst, which the lanes cannot hold.
    #[target_feature(enable = "avx512f,avx512ifma")]
    pub fn compute(&mut self, buckets: &Buckets, store: &PointStore) -> bool {
        self.sums.clear();
        for first_window in (0..buckets.windows()).step_by(LANES) {
            match window_group_sums(buckets, store, first_window) {
                Some(group) => {
                    self.sums.extend(&group[..LANES.min(buckets.windows() - first_window)])
                }
                None => return false,
            }
        }
        true
    }

    /// Σ_j 2^(w·j)·S_j.
    pub fn combine(&self, window_bits: usize) -> blst_p1 {
        let mut total = blst_p1::default();
        for sum in self.sums.iter().rev() {
            for _ in 0..window_bits {
                let doubled = total;
                // SAFETY: valid pointers to single values.
                unsafe { blst_p1_double(&mut total, &doubled) };
            }
            let before = total;
            // SAFETY: valid pointers to single values.
            unsafe { blst_p1_add_or_double(&mut total, &before, sum) };
        }
        total
    }
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn window_group_sums(
    buckets: &Buckets,
    store: &PointStore,
    first_window: usize,
) -> Option<[blst_p1; LANES]> {
    let mut t = G1x8::from_affine(&Fp8::zero(), &Fp8::zero());
    let mut s = G1x8::from_affine(&Fp8::zero(), &Fp8::zero());
    let mut t_infinite: __mmask8 = 0xff;
    let mut s_infinite: __mmask8 = 0xff;
    for bucket in (1..=buckets.buckets_per_window()).rev() {
        let mut index = [0i64; LANES];
        let mut present: __mmask8 = 0;
        let mut negated: __mmask8 = 0;
        for lane in 0..LANES {
            let window = first_window + lane;
            if window >= buckets.windows() {
                continue;
            }
            if let Some(reference) = buckets.first(window, bucket) {
                index[lane] = i64::from(reference & !NEGATED);
                present |= 1 << lane;
                if reference & NEGATED != 0 {
                    negated |= 1 << lane;
                }
            }
        }

        if present != 0 {
            let (bx, by) = store.gather(load_indices(&index));
            let by = Fp8::select(negated, &by, &by.neg());
            let (sum, undecided) = t.add_affine(&bx, &by);
            let first = present & t_infinite;
            let adding = present & !t_infinite;
            let mut next = G1x8 {
                x: Fp8::select(adding, &Fp8::select(first, &t.x, &bx), &sum.x),
                y: Fp8::select(adding, &Fp8::select(first, &t.y, &by), &sum.y),
                z: Fp8::select(adding, &Fp8::select(first, &t.z, &Fp8::one()), &sum.z),
            };
            let exceptional = undecided & adding;
            if exceptional != 0 {
                next = add_affine_in_blst(store, &t, (&bx, &by), next, exceptional)?;
            }
            t = next;
            t_infinite &= !present;
        }

        let (sum, undecided) = s.add(&t);
        let first = !t_infinite & s_infinite;
        let adding = !t_infinite & !s_infinite;
        let mut next = G1x8 {
            x: Fp8::select(adding, &Fp8::select(first, &s.x, &t.x), &sum.x),
            y: Fp8::select(adding, &Fp8::select(first, &s.y, &t.y), &sum.y),
            z: Fp8::select(adding, &Fp8::select(first, &s.z, &t.z), &sum.z),
        };
        let exceptional = undecided & adding;
        if exceptional != 0 {
            next = add_in_blst(store, &s, &t, next, exceptional)?;
        }
        s = next;
        s_infinite &= t_infinite;
    }

    let points = jacobian_lanes(&s);
    Some(std::array::from_fn(|lane| {
        if s_infinite & (1 << lane) != 0 { blst_p1::default() } else { points[lane] }
    }))
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn load_indices(indices: &[i64; LANES]) -> __m512i {
    // SAFETY: an unaligned load of the 64 bytes of `indices`.
    unsafe { _mm512_loadu_si512(indices.as_ptr().cast()) }
}

/// `next` with the `exceptional` lanes of `t + b` taken from blst's complete
/// addition. None when one is infinity, which the lanes cannot hold.
#[target_feature(enable = "avx512f,avx512ifma")]
fn add_affine_in_blst(
    store: &PointStore,
    t: &G1x8,
    b: (&Fp8, &Fp8),
    next: G1x8,
    exceptional: __mmask8,
) -> Option<G1x8> {
    let ts = jacobian_lanes(t);
    let bs = affine_lanes(b.0, b.1);
    let mut nexts = jacobian_lanes(&next);
    for lane in (0..LANES).filter(|lane| exceptional & (1 << lane) != 0) {
        let mut sum = blst_p1::default();
        // SAFETY: valid pointers to single values.
        unsafe { blst_p1_add_or_double_affine(&mut sum, &ts[lane], &bs[lane]) };
        // SAFETY: a valid pointer to a single value.
        if unsafe { blst_p1_is_inf(&sum) } {
            return None;
        }
        nexts[lane] = sum;
    }
    Some(jacobian_from_lanes(store, &nexts))
}

/// As `add_affine_in_blst`, for `s + t`.
#[target_feature(enable = "avx512f,avx512ifma")]
fn add_in_blst(
    store: &PointStore,
    s: &G1x8,
    t: &G1x8,
    next: G1x8,
    exceptional: __mmask8,
) -> Option<G1x8> {
    let ss = jacobian_lanes(s);
    let ts = jacobian_lanes(t);
    let mut nexts = jacobian_lanes(&next);
    for lane in (0..LANES).filter(|lane| exceptional & (1 << lane) != 0) {
        let mut sum = blst_p1::default();
        // SAFETY: valid pointers to single values.
        unsafe { blst_p1_add_or_double(&mut sum, &ss[lane], &ts[lane]) };
        // SAFETY: a valid pointer to a single value.
        if unsafe { blst_p1_is_inf(&sum) } {
            return None;
        }
        nexts[lane] = sum;
    }
    Some(jacobian_from_lanes(store, &nexts))
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn jacobian_lanes(point: &G1x8) -> [blst_p1; LANES] {
    let (x, y, z) = (point.x.to_blst_limbs(), point.y.to_blst_limbs(), point.z.to_blst_limbs());
    std::array::from_fn(|lane| blst_p1 {
        x: blst_fp { l: x[lane] },
        y: blst_fp { l: y[lane] },
        z: blst_fp { l: z[lane] },
    })
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn affine_lanes(x: &Fp8, y: &Fp8) -> [blst_p1_affine; LANES] {
    let (x, y) = (x.to_blst_limbs(), y.to_blst_limbs());
    std::array::from_fn(|lane| blst_p1_affine {
        x: blst_fp { l: x[lane] },
        y: blst_fp { l: y[lane] },
    })
}

#[target_feature(enable = "avx512f,avx512ifma")]
fn jacobian_from_lanes(store: &PointStore, points: &[blst_p1; LANES]) -> G1x8 {
    G1x8 {
        x: store.fp8_from_blst(&std::array::from_fn(|lane| points[lane].x.l)),
        y: store.fp8_from_blst(&std::array::from_fn(|lane| points[lane].y.l)),
        z: store.fp8_from_blst(&std::array::from_fn(|lane| points[lane].z.l)),
    }
}
