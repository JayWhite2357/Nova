//! This library implements Nova, a high-speed recursive SNARK.
#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![forbid(unsafe_code)]
#![cfg_attr(not(test), warn(clippy::print_stdout, clippy::print_stderr))]

// main APIs exposed by this library
#[cfg(feature = "parallel")]
pub mod nova;

#[cfg(feature = "experimental")]
pub mod neutron;

// public modules
pub mod errors;
pub mod frontend;
pub mod gadgets;
pub mod provider;
#[cfg(feature = "parallel")]
pub mod spartan;
pub mod traits;

// private modules
mod constants;
#[cfg(feature = "parallel")]
mod digest;
#[cfg(feature = "parallel")]
mod r1cs;

#[cfg(feature = "parallel")]
use traits::{commitment::CommitmentEngineTrait, Engine};

// some type aliases
#[cfg(feature = "parallel")]
type CommitmentKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey;
#[cfg(feature = "parallel")]
type DerandKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::DerandKey;
#[cfg(feature = "parallel")]
type Commitment<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment;
#[cfg(feature = "parallel")]
type CE<E> = <E as Engine>::CE;
