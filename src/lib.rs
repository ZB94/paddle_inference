#![doc = include_str!("../README.md")]

pub mod common;
pub mod config;
pub mod ctypes;
mod predictor;
mod tensor;
pub mod utils;

use libloading::{library_filename, Library};
use once_cell::sync::Lazy;
pub use predictor::Predictor;
pub use tensor::Tensor;

static LIBRARY: Lazy<Library> = Lazy::new(|| unsafe {
    Library::new(library_filename("paddle_inference_c")).expect("can't not load paddle_inference_c")
});

/// 用于快速调用实现了[`ctypes::Function`]的对象
///
/// **使用方法：**
///
/// ``` no_run
/// use paddle_inference::call;
/// use paddle_inference::ctypes::{Function, PD_ConfigCreate, PD_ConfigDestroy};
///
/// let ptr = call!(PD_ConfigCreate());
/// call!(PD_ConfigDestroy(ptr));
///
/// /// 等同于
/// let ptr = unsafe { PD_ConfigCreate::get()() };
/// unsafe { PD_ConfigDestroy::get()(ptr) };
/// ```
#[macro_export]
macro_rules! call {
    ($name: ident ( $( $args: expr ),* )) => {
        unsafe {
            use crate::ctypes::Function;
            <$name>::get()( $($args),* )
        }
    };
}
