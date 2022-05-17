use crate::common::OneDimArraySize;
use crate::ctypes::{PD_OneDimArraySize, PD_TwoDimArraySize, PD_TwoDimArraySizeDestroy};
use std::ops::Deref;

pub struct TwoDimArraySize {
    pub(crate) ptr: *mut PD_TwoDimArraySize,
    data: Option<(Vec<OneDimArraySize>, Vec<*mut PD_OneDimArraySize>)>,
}

impl TwoDimArraySize {
    pub fn from_ptr(ptr: *mut PD_TwoDimArraySize) -> Self {
        Self { ptr, data: None }
    }

    pub fn new<V: Into<Vec<OneDimArraySize>>>(v: V) -> Self {
        let data = v.into();
        let mut data_ptr = data.iter().map(|d| d.ptr).collect::<Vec<_>>();

        let size = data.len();
        let array = PD_TwoDimArraySize {
            size,
            data: data_ptr.as_mut_ptr(),
        };
        let ptr = Box::into_raw(Box::new(array));
        Self {
            ptr,
            data: Some((data, data_ptr)),
        }
    }
}

impl Drop for TwoDimArraySize {
    fn drop(&mut self) {
        if self.data.is_some() {
            unsafe { Box::from_raw(self.ptr) };
        } else {
            unsafe { PD_TwoDimArraySizeDestroy(self.ptr) }
        }
    }
}

impl Deref for TwoDimArraySize {
    type Target = [OneDimArraySize];

    fn deref(&self) -> &Self::Target {
        if let Some((data, _)) = &self.data {
            data.as_slice()
        } else {
            let ptr = unsafe { &*self.ptr };
            unsafe { std::slice::from_raw_parts(ptr.data as *const _, ptr.size) }
        }
    }
}
