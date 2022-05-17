use crate::ctypes::{PD_OneDimArrayCstr, PD_OneDimArrayCstrDestroy};
use std::borrow::Cow;
use std::ffi::{CStr, CString, NulError};
use std::os::raw::c_char;

pub struct OneDimArrayCstr {
    ptr: *mut PD_OneDimArrayCstr,
    data: Option<Vec<*const c_char>>,
}

impl OneDimArrayCstr {
    pub fn from_ptr(ptr: *mut PD_OneDimArrayCstr) -> Self {
        Self { ptr, data: None }
    }

    pub fn new<S: AsRef<str>, I: Iterator<Item = S>>(iter: I) -> Result<Self, NulError> {
        let mut data = iter
            .map(|s| CString::new(s.as_ref()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|s| s.into_raw() as *const _)
            .collect::<Vec<_>>();
        let ptr = {
            let array = PD_OneDimArrayCstr {
                size: data.len(),
                data: data.as_mut_ptr() as *mut _,
            };
            Box::into_raw(Box::new(array))
        };
        Ok(Self {
            ptr,
            data: Some(data),
        })
    }

    pub fn len(&self) -> usize {
        unsafe { (*self.ptr).size }
    }

    pub fn get(&self, idx: usize) -> Option<Cow<str>> {
        if idx >= self.len() {
            return None;
        }

        let s = if let Some(data) = &self.data {
            unsafe { CStr::from_ptr(data[idx]) }
        } else if idx <= isize::MAX as usize {
            unsafe { CStr::from_ptr((*self.ptr).data.offset(idx as isize).read()) }
        } else {
            return None;
        };

        Some(s.to_string_lossy())
    }
}

impl Drop for OneDimArrayCstr {
    fn drop(&mut self) {
        if let Some(data) = &mut self.data {
            unsafe { Box::from_raw(self.ptr) };
            for ptr in data {
                let ptr = *ptr;
                unsafe { CString::from_raw((*ptr) as *mut _) };
            }
        } else {
            unsafe { PD_OneDimArrayCstrDestroy(self.ptr) }
        }
    }
}

impl From<OneDimArrayCstr> for Vec<String> {
    fn from(array: OneDimArrayCstr) -> Self {
        (0..array.len())
            .filter_map(|idx| array.get(idx).map(|s| s.to_string()))
            .collect()
    }
}
