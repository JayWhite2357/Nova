//! This module implements various gadgets necessary for doing non-native arithmetic
//! Code in this module is adapted from [bellman-bignat](https://github.com/alex-ozdemir/bellman-bignat), which is licenced under MIT

#[cfg(feature = "parallel")]
use crate::frontend::SynthesisError;
#[cfg(feature = "parallel")]
use ff::PrimeField;

#[cfg(feature = "parallel")]
trait OptionExt<T> {
  fn grab(&self) -> Result<&T, SynthesisError>;
}

#[cfg(feature = "parallel")]
impl<T> OptionExt<T> for Option<T> {
  fn grab(&self) -> Result<&T, SynthesisError> {
    self.as_ref().ok_or(SynthesisError::AssignmentMissing)
  }
}

#[cfg(feature = "parallel")]
trait BitAccess {
  fn get_bit(&self, i: usize) -> Option<bool>;
}

#[cfg(feature = "parallel")]
impl<Scalar: PrimeField> BitAccess for Scalar {
  fn get_bit(&self, i: usize) -> Option<bool> {
    if i as u32 >= Scalar::NUM_BITS {
      return None;
    }

    let (byte_pos, bit_pos) = (i / 8, i % 8);
    let byte = self.to_repr().as_ref()[byte_pos];
    let bit = (byte >> bit_pos) & 1;
    Some(bit == 1)
  }
}

pub mod bignat;
pub mod util;
