use crate::call;
use crate::ctypes::{PD_OneDimArraySize, PD_OneDimArraySizeDestroy};
use std::ops::{Deref, DerefMut};

pub struct OneDimArraySize {
    pub(crate) ptr: *mut PD_OneDimArraySize,
    data: Option<Vec<usize>>,
}

impl OneDimArraySize {
    pub fn from_ptr(ptr: *mut PD_OneDimArraySize) -> Self {
        Self { ptr, data: None }
    }

    pub fn with_size(size: usize) -> Self {
        Self::new(vec![0; size])
    }

    pub fn new<V: Into<Vec<usize>>>(v: V) -> Self {
        let mut data = v.into();
        let size = data.len();
        let array = PD_OneDimArraySize {
            size,
            data: data.as_mut_ptr(),
        };
        let ptr = Box::into_raw(Box::new(array));
        Self {
            ptr,
            data: Some(data),
        }
    }
}

impl Drop for OneDimArraySize {
    fn drop(&mut self) {
        if self.data.is_some() {
            unsafe { Box::from_raw(self.ptr) };
        } else {
            call! { PD_OneDimArraySizeDestroy(self.ptr) }
        }
    }
}

impl Deref for OneDimArraySize {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        if let Some(data) = &self.data {
            data.as_slice()
        } else {
            let ptr = unsafe { &*self.ptr };
            unsafe { std::slice::from_raw_parts(ptr.data as *const _, ptr.size) }
        }
    }
}

impl DerefMut for OneDimArraySize {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if let Some(data) = &mut self.data {
            data.as_mut_slice()
        } else {
            let ptr = unsafe { &*self.ptr };
            unsafe { std::slice::from_raw_parts_mut(ptr.data, ptr.size) }
        }
    }
}

impl From<OneDimArraySize> for Vec<usize> {
    fn from(mut array: OneDimArraySize) -> Self {
        if let Some(data) = &mut array.data {
            let mut ret = vec![];
            std::mem::swap(data, &mut ret);
            ret
        } else {
            array.to_vec()
        }
    }
}
