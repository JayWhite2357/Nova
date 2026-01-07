//! This module implements various gadgets necessary for Nova and applications built with Nova.
#[cfg(feature = "parallel")]
pub(crate) mod ecc;
pub(crate) mod nonnative;
pub(crate) mod utils;
