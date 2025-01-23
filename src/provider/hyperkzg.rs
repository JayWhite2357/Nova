//! This module implements Nova's evaluation engine using `HyperKZG`, a KZG-based polynomial commitment for multilinear polynomials
//! HyperKZG is based on the transformation from univariate PCS to multilinear PCS in the Gemini paper (section 2.4.2 in <https://eprint.iacr.org/2022/420.pdf>).
//! However, there are some key differences:
//! (1) HyperKZG works with multilinear polynomials represented in evaluation form (rather than in coefficient form in Gemini's transformation).
//! This means that Spartan's polynomial IOP can use commit to its polynomials as-is without incurring any interpolations or FFTs.
//! (2) HyperKZG is specialized to use KZG as the univariate commitment scheme, so it includes several optimizations (both during the transformation of multilinear-to-univariate claims
//! and within the KZG commitment scheme implementation itself).
#![allow(non_snake_case)]
use crate::{
  errors::NovaError,
  provider::traits::{DlogGroup, PairingGroup},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    evaluation::EvaluationEngineTrait,
    AbsorbInROTrait, Engine, ROTrait, TranscriptEngineTrait, TranscriptReprTrait,
  },
};
use core::{
  marker::PhantomData,
  ops::{Add, Mul, MulAssign},
};
use ff::Field;
use itertools::izip;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Alias to points on G1 that are in preprocessed form
type G1Affine<E> = <<E as Engine>::GE as DlogGroup>::AffineGroupElement;

/// Alias to points on G1 that are in preprocessed form
type G2Affine<E> = <<<E as Engine>::GE as PairingGroup>::G2 as DlogGroup>::AffineGroupElement;

/// KZG commitment key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentKey<E: Engine>
where
  E::GE: PairingGroup,
{
  ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  h: <E::GE as DlogGroup>::AffineGroupElement,
  tau_H: <<E::GE as PairingGroup>::G2 as DlogGroup>::AffineGroupElement, // needed only for the verifier key
}

impl<E: Engine> Len for CommitmentKey<E>
where
  E::GE: PairingGroup,
{
  fn length(&self) -> usize {
    self.ck.len()
  }
}

/// A type that holds blinding generator
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerandKey<E: Engine>
where
  E::GE: DlogGroup,
{
  h: <E::GE as DlogGroup>::AffineGroupElement,
}

/// A KZG commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<E: Engine>
where
  E::GE: PairingGroup,
{
  comm: <E as Engine>::GE,
}

impl<E: Engine> CommitmentTrait<E> for Commitment<E>
where
  E::GE: PairingGroup,
{
  fn to_coordinates(&self) -> (E::Base, E::Base, bool) {
    self.comm.to_coordinates()
  }
}

impl<E: Engine> Default for Commitment<E>
where
  E::GE: PairingGroup,
{
  fn default() -> Self {
    Commitment {
      comm: E::GE::zero(),
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for Commitment<E>
where
  E::GE: PairingGroup,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    let is_infinity_byte = (!is_infinity).into();
    [
      x.to_transcript_bytes(),
      y.to_transcript_bytes(),
      [is_infinity_byte].to_vec(),
    ]
    .concat()
  }
}

impl<E: Engine> AbsorbInROTrait<E> for Commitment<E>
where
  E::GE: PairingGroup,
{
  fn absorb_in_ro(&self, ro: &mut E::RO) {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    ro.absorb(x);
    ro.absorb(y);
    ro.absorb(if is_infinity {
      E::Base::ONE
    } else {
      E::Base::ZERO
    });
  }
}

impl<E: Engine> MulAssign<E::Scalar> for Commitment<E>
where
  E::GE: PairingGroup,
{
  fn mul_assign(&mut self, scalar: E::Scalar) {
    let result = (self as &Commitment<E>).comm * scalar;
    *self = Commitment { comm: result };
  }
}

impl<'b, E: Engine> Mul<&'b E::Scalar> for &'_ Commitment<E>
where
  E::GE: PairingGroup,
{
  type Output = Commitment<E>;

  fn mul(self, scalar: &'b E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E: Engine> Mul<E::Scalar> for Commitment<E>
where
  E::GE: PairingGroup,
{
  type Output = Commitment<E>;

  fn mul(self, scalar: E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E: Engine> Add for Commitment<E>
where
  E::GE: PairingGroup,
{
  type Output = Commitment<E>;

  fn add(self, other: Commitment<E>) -> Commitment<E> {
    Commitment {
      comm: self.comm + other.comm,
    }
  }
}

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> CommitmentEngineTrait<E> for CommitmentEngine<E>
where
  E::GE: PairingGroup,
{
  type Commitment = Commitment<E>;
  type CommitmentKey = CommitmentKey<E>;
  type DerandKey = DerandKey<E>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    // NOTE: this is for testing purposes and should not be used in production
    // TODO: we need to decide how to generate load/store parameters
    let tau = E::Scalar::random(OsRng);
    let num_gens = n.next_power_of_two();

    // Compute powers of tau in E::Scalar, then scalar muls in parallel
    let mut powers_of_tau: Vec<E::Scalar> = Vec::with_capacity(num_gens);
    powers_of_tau.insert(0, E::Scalar::ONE);
    for i in 1..num_gens {
      powers_of_tau.insert(i, powers_of_tau[i - 1] * tau);
    }

    let ck: Vec<G1Affine<E>> = (0..num_gens)
      .into_par_iter()
      .map(|i| (<E::GE as DlogGroup>::gen() * powers_of_tau[i]).affine())
      .collect();

    let h = E::GE::from_label(label, 1).first().unwrap().clone();

    let tau_H = (<<E::GE as PairingGroup>::G2 as DlogGroup>::gen() * tau).affine();

    Self::CommitmentKey { ck, h, tau_H }
  }

  fn derand_key(ck: &Self::CommitmentKey) -> Self::DerandKey {
    Self::DerandKey { h: ck.h.clone() }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar], r: &E::Scalar) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());

    let mut scalars: Vec<E::Scalar> = v.to_vec();
    scalars.push(*r);
    let mut bases = ck.ck[..v.len()].to_vec();
    bases.push(ck.h.clone());

    Commitment {
      comm: E::GE::vartime_multiscalar_mul(&scalars, &bases),
    }
  }

  fn derandomize(
    dk: &Self::DerandKey,
    commit: &Self::Commitment,
    r: &E::Scalar,
  ) -> Self::Commitment {
    Commitment {
      comm: commit.comm - <E::GE as DlogGroup>::group(&dk.h) * r,
    }
  }
}

/// Provides an implementation of generators for proving evaluations
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine> {
  _p: PhantomData<E>,
}

/// A verifier key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine>
where
  E::GE: PairingGroup,
{
  G: G1Affine<E>,
  H: G2Affine<E>,
  tau_H: G2Affine<E>,
}

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct EvaluationArgument<E: Engine>
where
  E::GE: PairingGroup,
{
  com: Vec<G1Affine<E>>,
  w: [G1Affine<E>; 3],
  v: [Vec<E::Scalar>; 3],
}

/// Provides an implementation of a polynomial evaluation engine using KZG
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> EvaluationEngine<E>
where
  E::GE: PairingGroup,
{
  // This impl block defines helper functions that are not a part of
  // EvaluationEngineTrait, but that we will use to implement the trait methods.
  fn compute_challenge(com: &[G1Affine<E>], transcript: &mut <E as Engine>::TE) -> E::Scalar {
    transcript.absorb(b"c", &com.to_vec().as_slice());

    transcript.squeeze(b"c").unwrap()
  }

  // Compute challenge q = Hash(vk, C0, ..., C_{k-1}, u0, ...., u_{t-1},
  // (f_i(u_j))_{i=0..k-1,j=0..t-1})
  fn get_batch_challenge(v: &[Vec<E::Scalar>], transcript: &mut <E as Engine>::TE) -> E::Scalar {
    transcript.absorb(
      b"v",
      &v.iter()
        .flatten()
        .cloned()
        .collect::<Vec<E::Scalar>>()
        .as_slice(),
    );

    transcript.squeeze(b"r").unwrap()
  }

  fn batch_challenge_powers(q: E::Scalar, k: usize) -> Vec<E::Scalar> {
    // Compute powers of q : (1, q, q^2, ..., q^(k-1))
    let mut q_powers = vec![E::Scalar::ONE; k];
    for i in 1..k {
      q_powers[i] = q_powers[i - 1] * q;
    }
    q_powers
  }

  fn verifier_second_challenge(W: &[G1Affine<E>], transcript: &mut <E as Engine>::TE) -> E::Scalar {
    transcript.absorb(b"W", &W.to_vec().as_slice());

    transcript.squeeze(b"d").unwrap()
  }
}

impl<E> EvaluationEngineTrait<E> for EvaluationEngine<E>
where
  E: Engine<CE = CommitmentEngine<E>>,
  E::GE: PairingGroup,
{
  type EvaluationArgument = EvaluationArgument<E>;
  type ProverKey = ProverKey<E>;
  type VerifierKey = VerifierKey<E>;

  fn setup(
    ck: &<E::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    let pk = ProverKey {
      _p: Default::default(),
    };

    let vk = VerifierKey {
      G: E::GE::gen().affine(),
      H: <<E::GE as PairingGroup>::G2 as DlogGroup>::gen().affine(),
      tau_H: ck.tau_H.clone(),
    };

    (pk, vk)
  }

  fn prove(
    ck: &CommitmentKey<E>,
    _pk: &Self::ProverKey,
    transcript: &mut <E as Engine>::TE,
    _C: &Commitment<E>,
    hat_P: &[E::Scalar],
    point: &[E::Scalar],
    _eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let x: Vec<E::Scalar> = point.to_vec();

    //////////////// begin helper closures //////////
    let kzg_open = |f: &[E::Scalar], u: E::Scalar| -> G1Affine<E> {
      // On input f(x) and u compute the witness polynomial used to prove
      // that f(u) = v. The main part of this is to compute the
      // division (f(x) - f(u)) / (x - u), but we don't use a general
      // division algorithm, we make use of the fact that the division
      // never has a remainder, and that the denominator is always a linear
      // polynomial. The cost is (d-1) mults + (d-1) adds in E::Scalar, where
      // d is the degree of f.
      //
      // We use the fact that if we compute the quotient of f(x)/(x-u),
      // there will be a remainder, but it'll be v = f(u).  Put another way
      // the quotient of f(x)/(x-u) and (f(x) - f(v))/(x-u) is the
      // same.  One advantage is that computing f(u) could be decoupled
      // from kzg_open, it could be done later or separate from computing W.

      let compute_witness_polynomial = |f: &[E::Scalar], u: E::Scalar| -> Vec<E::Scalar> {
        let d = f.len();

        // Compute h(x) = f(x)/(x - u)
        let mut h = vec![E::Scalar::ZERO; d];
        for i in (1..d).rev() {
          h[i - 1] = f[i] + h[i] * u;
        }

        h
      };

      let h = compute_witness_polynomial(f, u);

      E::CE::commit(ck, &h, &E::Scalar::ZERO).comm.affine()
    };

    let kzg_open_batch = |f: &[Vec<E::Scalar>],
                          u: &[E::Scalar],
                          transcript: &mut <E as Engine>::TE|
     -> (Vec<G1Affine<E>>, Vec<Vec<E::Scalar>>) {
      let poly_eval = |f: &[E::Scalar], u: E::Scalar| -> E::Scalar {
        let mut v = f[0];
        let mut u_power = E::Scalar::ONE;

        for fi in f.iter().skip(1) {
          u_power *= u;
          v += u_power * fi;
        }

        v
      };

      let scalar_vector_muladd = |a: &mut Vec<E::Scalar>, v: &Vec<E::Scalar>, s: E::Scalar| {
        assert!(a.len() >= v.len());
        for i in 0..v.len() {
          a[i] += s * v[i];
        }
      };

      let kzg_compute_batch_polynomial = |f: &[Vec<E::Scalar>], q: E::Scalar| -> Vec<E::Scalar> {
        let k = f.len(); // Number of polynomials we're batching

        let q_powers = Self::batch_challenge_powers(q, k);

        // Compute B(x) = f[0] + q*f[1] + q^2 * f[2] + ... q^(k-1) * f[k-1]
        let mut B = f[0].clone();
        for i in 1..k {
          scalar_vector_muladd(&mut B, &f[i], q_powers[i]); // B += q_powers[i] * f[i]
        }

        B
      };
      ///////// END kzg_open_batch closure helpers

      let k = f.len();
      let t = u.len();

      // The verifier needs f_i(u_j), so we compute them here
      // (V will compute B(u_j) itself)
      let mut v = vec![vec!(E::Scalar::ZERO; k); t];
      v.par_iter_mut().enumerate().for_each(|(i, v_i)| {
        // for each point u
        v_i.par_iter_mut().zip_eq(f).for_each(|(v_ij, f)| {
          // for each poly f
          // for each poly f (except the last one - since it is constant)
          *v_ij = poly_eval(f, u[i]);
        });
      });

      let q = Self::get_batch_challenge(&v, transcript);
      let B = kzg_compute_batch_polynomial(f, q);

      // Now open B at u0, ..., u_{t-1}
      let w = u
        .into_par_iter()
        .map(|ui| kzg_open(&B, *ui))
        .collect::<Vec<G1Affine<E>>>();

      // The prover computes the challenge to keep the transcript in the same
      // state as that of the verifier
      let _d_0 = Self::verifier_second_challenge(&w, transcript);

      (w, v)
    };

    ///// END helper closures //////////

    let ell = x.len();
    let n = hat_P.len();
    assert_eq!(n, 1 << ell); // Below we assume that n is a power of two

    // Phase 1  -- create commitments com_1, ..., com_\ell
    // We do not compute final Pi (and its commitment) as it is constant and equals to 'eval'
    // also known to verifier, so can be derived on its side as well
    let mut polys: Vec<Vec<E::Scalar>> = Vec::new();
    polys.push(hat_P.to_vec());
    for i in 0..ell - 1 {
      let Pi_len = polys[i].len() / 2;
      let mut Pi = vec![E::Scalar::ZERO; Pi_len];

      #[allow(clippy::needless_range_loop)]
      Pi.par_iter_mut().enumerate().for_each(|(j, Pi_j)| {
        *Pi_j = x[ell - i - 1] * (polys[i][2 * j + 1] - polys[i][2 * j]) + polys[i][2 * j];
      });

      polys.push(Pi);
    }

    // We do not need to commit to the first polynomial as it is already committed.
    // Compute commitments in parallel
    let com: Vec<G1Affine<E>> = (1..polys.len())
      .into_par_iter()
      .map(|i| E::CE::commit(ck, &polys[i], &E::Scalar::ZERO).comm.affine())
      .collect();

    // Phase 2
    // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
    // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
    let r = Self::compute_challenge(&com, transcript);
    let u = vec![r, -r, r * r];

    // Phase 3 -- create response
    let (w, v) = kzg_open_batch(&polys, &u, transcript);

    Ok(EvaluationArgument {
      com,
      w: w.try_into().expect("w should have length 3"),
      v: v.try_into().expect("v should have length 3"),
    })
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    VerifierKey { G, H, tau_H }: &Self::VerifierKey,
    transcript: &mut <E as Engine>::TE,
    &Commitment { comm: commitment }: &Commitment<E>,
    x: &[E::Scalar],
    y: &E::Scalar,
    EvaluationArgument { com, w, v }: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let r = Self::compute_challenge(com, transcript);
    let q = Self::get_batch_challenge(v, transcript);
    let d = Self::verifier_second_challenge(w, transcript);

    if r == E::Scalar::ZERO
      || commitment == E::GE::zero()
      || v[0].len() != x.len()
      || v[1].len() != x.len()
      || v[2].len() != x.len()
      || com.len() != x.len() - 1
    {
      return Err(NovaError::ProofVerifyError);
    }

    let mut last_v2 = *y;
    let mut b0 = E::Scalar::ZERO;
    let mut b1 = E::Scalar::ZERO;
    let mut b2 = E::Scalar::ZERO;

    for ((&v0i, &v1i, &v2i), &xi) in izip!(izip!(&v[0], &v[1], &v[2]).rev(), x) {
      if r * (last_v2.double() + (xi - E::Scalar::ONE) * (v0i + v1i)) + xi * (v1i - v0i)
        != E::Scalar::ZERO
      {
        return Err(NovaError::ProofVerifyError);
      }
      b0 *= q;
      b1 *= q;
      b2 *= q;
      b0 += v0i;
      b1 += v1i;
      b2 += v2i;
      last_v2 = v2i;
    }
    b2 *= d;
    b2 += b1;
    b2 *= d;
    b2 += b0;

    let mut g_l = com.last().map(E::GE::group).unwrap_or(E::GE::zero());
    for c in com.iter().rev().skip(1) {
      g_l *= q;
      g_l += E::GE::group(c);
    }
    g_l *= q;
    g_l += commitment;
    g_l *= E::Scalar::ONE + d * (E::Scalar::ONE + d);
    g_l += E::GE::group(&w[0]) * r;
    g_l += E::GE::group(&w[1]) * -d * r;
    g_l += E::GE::group(&w[2]) * (d * r).square();
    g_l += E::GE::group(G) * -b2;

    let mut g_r = E::GE::group(&w[2]);
    g_r *= d;
    g_r += E::GE::group(&w[1]);
    g_r *= d;
    g_r += E::GE::group(&w[0]);

    if (E::GE::pairing(&g_l, &DlogGroup::group(H)))
      != (E::GE::pairing(&g_r, &DlogGroup::group(tau_H)))
    {
      return Err(NovaError::ProofVerifyError);
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::{keccak::Keccak256Transcript, Bn256EngineKZG},
    spartan::polys::multilinear::MultilinearPolynomial,
  };
  use bincode::Options;
  use rand::SeedableRng;

  type E = Bn256EngineKZG;
  type Fr = <E as Engine>::Scalar;

  #[test]
  fn test_hyperkzg_eval() {
    // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
    let n = 4;
    let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);
    let (pk, vk): (ProverKey<E>, VerifierKey<E>) = EvaluationEngine::setup(&ck);

    // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
    let poly = vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4)];

    let C = CommitmentEngine::commit(&ck, &poly, &<E as Engine>::Scalar::ZERO);

    let test_inner = |point: Vec<Fr>, eval: Fr| -> Result<(), NovaError> {
      let mut tr = Keccak256Transcript::new(b"TestEval");
      let proof = EvaluationEngine::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).unwrap();
      let mut tr = Keccak256Transcript::new(b"TestEval");
      EvaluationEngine::verify(&vk, &mut tr, &C, &point, &eval, &proof)
    };

    // Call the prover with a (point, eval) pair.
    // The prover does not recompute so it may produce a proof, but it should not verify
    let point = vec![Fr::from(0), Fr::from(0)];
    let eval = Fr::ONE;
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(0), Fr::from(1)];
    let eval = Fr::from(2);
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(1), Fr::from(1)];
    let eval = Fr::from(4);
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(3);
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(9);
    assert!(test_inner(point, eval).is_ok());

    // Try a couple incorrect evaluations and expect failure
    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(50);
    assert!(test_inner(point, eval).is_err());

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(4);
    assert!(test_inner(point, eval).is_err());
  }

  #[test]
  fn test_hyperkzg_small() {
    let n = 4;

    // poly = [1, 2, 1, 4]
    let poly = vec![Fr::ONE, Fr::from(2), Fr::from(1), Fr::from(4)];

    // point = [4,3]
    let point = vec![Fr::from(4), Fr::from(3)];

    // eval = 28
    let eval = Fr::from(28);

    let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);
    let (pk, vk) = EvaluationEngine::setup(&ck);

    // make a commitment
    let C = CommitmentEngine::commit(&ck, &poly, &<E as Engine>::Scalar::ZERO);

    // prove an evaluation
    let mut prover_transcript = Keccak256Transcript::new(b"TestEval");
    let proof =
      EvaluationEngine::<E>::prove(&ck, &pk, &mut prover_transcript, &C, &poly, &point, &eval)
        .unwrap();
    let post_c_p = prover_transcript.squeeze(b"c").unwrap();

    // verify the evaluation
    let mut verifier_transcript = Keccak256Transcript::new(b"TestEval");
    assert!(
      EvaluationEngine::verify(&vk, &mut verifier_transcript, &C, &point, &eval, &proof).is_ok()
    );
    let post_c_v = verifier_transcript.squeeze(b"c").unwrap();

    // check if the prover transcript and verifier transcript are kept in the same state
    assert_eq!(post_c_p, post_c_v);

    let proof_bytes = bincode::DefaultOptions::new()
      .with_big_endian()
      .with_fixint_encoding()
      .serialize(&proof)
      .unwrap();
    assert_eq!(proof_bytes.len(), 352);

    // Change the proof and expect verification to fail
    let mut bad_proof = proof.clone();
    let v1 = bad_proof.v[1].clone();
    bad_proof.v[0].clone_from(&v1);
    let mut verifier_transcript2 = Keccak256Transcript::new(b"TestEval");
    assert!(EvaluationEngine::verify(
      &vk,
      &mut verifier_transcript2,
      &C,
      &point,
      &eval,
      &bad_proof
    )
    .is_err());
  }

  #[test]
  fn test_hyperkzg_large() {
    // test the hyperkzg prover and verifier with random instances (derived from a seed)
    for ell in [4, 5, 6] {
      let mut rng = rand::rngs::StdRng::seed_from_u64(ell as u64);

      let n = 1 << ell; // n = 2^ell

      let poly = (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
      let point = (0..ell).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
      let eval = MultilinearPolynomial::evaluate_with(&poly, &point);

      let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);
      let (pk, vk) = EvaluationEngine::setup(&ck);

      // make a commitment
      let C = CommitmentEngine::commit(&ck, &poly, &<E as Engine>::Scalar::ZERO);

      // prove an evaluation
      let mut prover_transcript = Keccak256Transcript::new(b"TestEval");
      let proof: EvaluationArgument<E> =
        EvaluationEngine::prove(&ck, &pk, &mut prover_transcript, &C, &poly, &point, &eval)
          .unwrap();

      // verify the evaluation
      let mut verifier_tr = Keccak256Transcript::new(b"TestEval");
      assert!(EvaluationEngine::verify(&vk, &mut verifier_tr, &C, &point, &eval, &proof).is_ok());

      // Change the proof and expect verification to fail
      let mut bad_proof = proof.clone();
      let v1 = bad_proof.v[1].clone();
      bad_proof.v[0].clone_from(&v1);
      let mut verifier_tr2 = Keccak256Transcript::new(b"TestEval");
      assert!(
        EvaluationEngine::verify(&vk, &mut verifier_tr2, &C, &point, &eval, &bad_proof).is_err()
      );
    }
  }
}
