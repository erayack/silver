//! `CellBatchVerifier` on AVX-512 IFMA: the same verdicts, with the scalar
//! side and the multi-scalar multiplications on vectors.

#[cfg(target_arch = "x86_64")]
pub use x86::SimdCellBatchVerifier;

#[cfg(not(target_arch = "x86_64"))]
pub struct SimdCellBatchVerifier;

#[cfg(not(target_arch = "x86_64"))]
impl SimdCellBatchVerifier {
    pub fn new() -> Option<Self> {
        None
    }

    pub fn verify(
        &mut self,
        _commitments: &[[u8; crate::bls::COMPRESSED_LEN]],
        _cell_indices: &[u64],
        _cells: &[[u8; crate::kzg::BYTES_PER_CELL]],
        _proofs: &[[u8; crate::bls::COMPRESSED_LEN]],
    ) -> bool {
        unreachable!("`new` is None without x86_64")
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use blst::{blst_fr, blst_p1};

    use crate::{
        bls::{COMPRESSED_LEN, blst_p1_affine, simd_available, uncompress_in_g1},
        kzg::{
            BYTES_PER_CELL, BYTES_PER_FIELD_ELEMENT, CELLS_PER_EXT_BLOB, FIELD_ELEMENTS_PER_CELL,
            blst_ops::{
                LincombScratch, fr_add, fr_from_bytes, fr_from_u64, fr_mul, g1_add, g1_lincomb,
                g1_neg, g1_to_affine,
            },
            challenge::Preimage,
            cosets::Cosets,
            domain::reverse_bits,
            fft::inverse_fft,
            fixed_base::{FixedBase, scalar_bytes},
            pairing_lines::PairingLines,
            setup::Setup,
            verify::Deduplicated,
        },
    };

    const LOG_CELL: u32 = 6;

    pub struct SimdCellBatchVerifier {
        setup: Setup,
        cosets: Cosets,
        commitments: Deduplicated,
        decoded: Vec<Option<blst_p1_affine>>,
        commitment_points: Vec<blst_p1_affine>,
        proof_points: Vec<blst_p1_affine>,
        evaluations: Vec<blst_fr>,
        preimage: Preimage,
        r_powers: Vec<blst_fr>,
        scalars: Vec<blst_fr>,
        columns: Vec<blst_fr>,
        column_used: Vec<bool>,
        column: Vec<blst_fr>,
        coefficients: Vec<blst_fr>,
        interpolation: Vec<blst_fr>,
        lincomb: LincombScratch,
        fixed_base: FixedBase,
        pairing_lines: PairingLines,
    }

    impl SimdCellBatchVerifier {
        /// None without AVX-512 IFMA.
        pub fn new() -> Option<Self> {
            if !simd_available() {
                return None;
            }
            let setup = Setup::decode();
            let fixed_base = FixedBase::new(setup.g1_monomial.as_slice().try_into().unwrap());
            let pairing_lines = PairingLines::new(&setup.g2_s_pow_cell, &setup.g2_neg_generator);
            Some(Self {
                setup,
                pairing_lines,
                fixed_base,
                cosets: Cosets::new(),
                commitments: Deduplicated::default(),
                decoded: Vec::new(),
                commitment_points: Vec::new(),
                proof_points: Vec::new(),
                evaluations: Vec::new(),
                preimage: Preimage::new(),
                r_powers: Vec::new(),
                scalars: Vec::new(),
                columns: Vec::new(),
                column_used: Vec::new(),
                column: Vec::new(),
                coefficients: Vec::new(),
                interpolation: Vec::new(),
                lincomb: LincombScratch::new(),
            })
        }

        /// True exactly when `CellBatchVerifier::verify` is.
        pub fn verify(
            &mut self,
            commitments: &[[u8; COMPRESSED_LEN]],
            cell_indices: &[u64],
            cells: &[[u8; BYTES_PER_CELL]],
            proofs: &[[u8; COMPRESSED_LEN]],
        ) -> bool {
            let n = commitments.len();
            if cell_indices.len() != n || cells.len() != n || proofs.len() != n {
                return false;
            }
            if cell_indices.iter().any(|&index| index >= CELLS_PER_EXT_BLOB as u64) {
                return false;
            }
            self.commitments.fill(commitments);
            if !decode_points(
                &self.commitments.unique,
                &mut self.decoded,
                &mut self.commitment_points,
            ) || !self.decode_cells(cells) ||
                !decode_points(proofs, &mut self.decoded, &mut self.proof_points)
            {
                return false;
            }
            let r = self.preimage.challenge(
                &self.commitments.unique,
                &self.commitments.index_of_cell,
                cell_indices,
                cells,
                proofs,
            );
            self.pairing_holds(&r, cell_indices)
        }

        fn decode_cells(&mut self, cells: &[[u8; BYTES_PER_CELL]]) -> bool {
            self.evaluations.clear();
            for cell in cells {
                for element in cell.chunks_exact(BYTES_PER_FIELD_ELEMENT) {
                    match fr_from_bytes(element.try_into().unwrap()) {
                        Some(value) => self.evaluations.push(value),
                        None => return false,
                    }
                }
            }
            true
        }

        /// e(LL, [s^64])·e(RL, −[1]) = 1, with LL = Σ r^k·π_k and RL = RLC −
        /// RLI + RLP.
        fn pairing_holds(&mut self, r: &blst_fr, cell_indices: &[u64]) -> bool {
            let n = cell_indices.len();
            let zero = fr_from_u64(0);
            self.r_powers.clear();
            let mut power = fr_from_u64(1);
            for _ in 0..n {
                self.r_powers.push(power);
                power = fr_mul(&power, r);
            }
            let ll = g1_lincomb(&self.proof_points, &self.r_powers, &mut self.lincomb);

            self.scalars.clear();
            self.scalars.resize(self.commitments.unique.len(), zero);
            for (k, &i) in self.commitments.index_of_cell.iter().enumerate() {
                self.scalars[i] = fr_add(&self.scalars[i], &self.r_powers[k]);
            }
            let rlc = g1_lincomb(&self.commitment_points, &self.scalars, &mut self.lincomb);

            let rli = self.interpolation_commitment(cell_indices);

            self.scalars.clear();
            for (k, &c) in cell_indices.iter().enumerate() {
                let shift = &self.cosets.shift_pow_cell[c as usize];
                self.scalars.push(fr_mul(&self.r_powers[k], shift));
            }
            let rlp = g1_lincomb(&self.proof_points, &self.scalars, &mut self.lincomb);

            let rl = g1_add(&g1_add(&rlc, &g1_neg(&rli)), &rlp);
            self.pairing_lines.check(&g1_to_affine(&ll), &g1_to_affine(&rl))
        }

        /// RLI = [Σ_k r^k·I_k(s)]. The r^k-weighted cells of each column are
        /// summed, put in natural order, inverse-transformed, and shifted from
        /// the column's coset by h⁻¹.
        fn interpolation_commitment(&mut self, cell_indices: &[u64]) -> blst_p1 {
            let zero = fr_from_u64(0);
            self.columns.clear();
            self.columns.resize(CELLS_PER_EXT_BLOB * FIELD_ELEMENTS_PER_CELL, zero);
            self.column_used.clear();
            self.column_used.resize(CELLS_PER_EXT_BLOB, false);
            for (k, &c) in cell_indices.iter().enumerate() {
                let c = c as usize;
                self.column_used[c] = true;
                for j in 0..FIELD_ELEMENTS_PER_CELL {
                    let term = fr_mul(
                        &self.r_powers[k],
                        &self.evaluations[k * FIELD_ELEMENTS_PER_CELL + j],
                    );
                    let at = c * FIELD_ELEMENTS_PER_CELL + j;
                    self.columns[at] = fr_add(&self.columns[at], &term);
                }
            }

            self.interpolation.clear();
            self.interpolation.resize(FIELD_ELEMENTS_PER_CELL, zero);
            self.column.resize(FIELD_ELEMENTS_PER_CELL, zero);
            self.coefficients.resize(FIELD_ELEMENTS_PER_CELL, zero);
            for c in (0..CELLS_PER_EXT_BLOB).filter(|&c| self.column_used[c]) {
                for j in 0..FIELD_ELEMENTS_PER_CELL {
                    self.column[j] =
                        self.columns[c * FIELD_ELEMENTS_PER_CELL + reverse_bits(j, LOG_CELL)];
                }
                inverse_fft(
                    &mut self.coefficients,
                    &self.column,
                    &self.cosets.inverse_roots,
                    &self.cosets.inverse_size,
                );
                let mut factor = fr_from_u64(1);
                for i in 0..FIELD_ELEMENTS_PER_CELL {
                    let term = fr_mul(&self.coefficients[i], &factor);
                    self.interpolation[i] = fr_add(&self.interpolation[i], &term);
                    factor = fr_mul(&factor, &self.cosets.shift_inverse[c]);
                }
            }
            let scalars = std::array::from_fn(|i| scalar_bytes(&self.interpolation[i]));
            self.fixed_base.mult(&scalars)
        }
    }

    /// Decodes `encodings` into `points`; false if any is not a valid G1 point.
    fn decode_points(
        encodings: &[[u8; COMPRESSED_LEN]],
        decoded: &mut Vec<Option<blst_p1_affine>>,
        points: &mut Vec<blst_p1_affine>,
    ) -> bool {
        decoded.clear();
        decoded.resize(encodings.len(), None);
        uncompress_in_g1(encodings, decoded);
        points.clear();
        for point in decoded.iter() {
            match point {
                Some(point) => points.push(*point),
                None => return false,
            }
        }
        true
    }
}
