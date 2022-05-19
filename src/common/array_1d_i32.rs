use crate::call;
use crate::ctypes::{PD_OneDimArrayInt32, PD_OneDimArrayInt32Destroy};
use std::ops::{Deref, DerefMut};

pub struct OneDimArrayInt32 {
    ptr: *mut PD_OneDimArrayInt32,
    data: Option<Vec<i32>>,
}

impl OneDimArrayInt32 {
    pub fn from_ptr(ptr: *mut PD_OneDimArrayInt32) -> Self {
        Self { ptr, data: None }
    }

    pub fn with_size(size: usize) -> Self {
        Self::new(vec![0i32; size])
    }

    pub fn new<V: Into<Vec<i32>>>(v: V) -> Self {
        let mut data = v.into();
        let size = data.len();
        let array = PD_OneDimArrayInt32 {
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

impl Drop for OneDimArrayInt32 {
    fn drop(&mut self) {
        if self.data.is_some() {
            unsafe { Box::from_raw(self.ptr) };
        } else {
            call! { PD_OneDimArrayInt32Destroy(self.ptr) }
        }
    }
}

impl Deref for OneDimArrayInt32 {
    type Target = [i32];

    fn deref(&self) -> &Self::Target {
        if let Some(data) = &self.data {
            data.as_slice()
        } else {
            let ptr = unsafe { &*self.ptr };
            unsafe { std::slice::from_raw_parts(ptr.data as *const _, ptr.size) }
        }
    }
}

impl DerefMut for OneDimArrayInt32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if let Some(data) = &mut self.data {
            data.as_mut_slice()
        } else {
            let ptr = unsafe { &*self.ptr };
            unsafe { std::slice::from_raw_parts_mut(ptr.data, ptr.size) }
        }
    }
}

impl From<OneDimArrayInt32> for Vec<i32> {
    fn from(mut array: OneDimArrayInt32) -> Self {
        if let Some(data) = &mut array.data {
            let mut ret = vec![];
            std::mem::swap(data, &mut ret);
            ret
        } else {
            array.to_vec()
        }
    }
}

#[test]
fn test() {
    let arr = OneDimArrayInt32::with_size(100);
    let ptr = unsafe { &mut *arr.ptr };
    unsafe { ptr.data.offset(50).write(10) };
    assert_eq!(arr[50], 10);
}
