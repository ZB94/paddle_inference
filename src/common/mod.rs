pub use crate::ctypes::{DataType, PlaceType, PrecisionType};
mod array_1d_cstr;
mod array_1d_i32;
mod array_1d_size;
mod array_2d_size;

pub use array_1d_cstr::OneDimArrayCstr;
pub use array_1d_i32::OneDimArrayInt32;
pub use array_1d_size::OneDimArraySize;
pub use array_2d_size::TwoDimArraySize;
