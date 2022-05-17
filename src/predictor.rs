use crate::common::OneDimArrayCstr;
use crate::ctypes::{
    PD_Predictor, PD_PredictorClone, PD_PredictorDestroy, PD_PredictorGetInputHandle,
    PD_PredictorGetInputNames, PD_PredictorGetInputNum, PD_PredictorGetOutputHandle,
    PD_PredictorGetOutputNames, PD_PredictorGetOutputNum, PD_PredictorRun,
};
use crate::tensor::Tensor;
use std::ffi::CString;

pub struct Predictor {
    ptr: *mut PD_Predictor,
}

/// Paddle Inference 的预测器
impl Predictor {
    pub fn from_ptr(ptr: *mut PD_Predictor) -> Self {
        Self { ptr }
    }
}

impl Predictor {
    /// 获取输入 Tensor 名称
    pub fn input_names(&self) -> OneDimArrayCstr {
        let ptr = unsafe { PD_PredictorGetInputNames(self.ptr) };
        OneDimArrayCstr::from_ptr(ptr)
    }

    /// 获取输入 Tensor 数量
    pub fn input_num(&self) -> usize {
        unsafe { PD_PredictorGetInputNum(self.ptr) }
    }

    /// 根据名称获取输入 Tensor
    ///
    /// **注意:** 如果输入名称中包含字符`\0`，则只会将`\0`之前的字符作为输入
    pub fn input(&self, name: &str) -> Tensor {
        let (_, name) = Self::to_c_str(name);
        let ptr = unsafe { PD_PredictorGetInputHandle(self.ptr, name) };
        Tensor::from_ptr(ptr)
    }

    /// 获取输出 Tensor 名称
    pub fn output_names(&self) -> OneDimArrayCstr {
        let ptr = unsafe { PD_PredictorGetOutputNames(self.ptr) };
        OneDimArrayCstr::from_ptr(ptr)
    }

    /// 获取输出 Tensor 数量
    pub fn output_num(&self) -> usize {
        unsafe { PD_PredictorGetOutputNum(self.ptr) }
    }

    /// 根据名称获取输出 Tensor
    ///
    /// **注意:** 如果输入名称中包含字符`\0`，则只会将`\0`之前的字符作为输入
    pub fn output(&self, name: &str) -> Tensor {
        let (_, name) = Self::to_c_str(name);
        let ptr = unsafe { PD_PredictorGetOutputHandle(self.ptr, name) };
        Tensor::from_ptr(ptr)
    }

    fn to_c_str(s: &str) -> (Option<CString>, *const i8) {
        CString::new(s)
            .map(|s| {
                let ptr = s.as_ptr();
                (Some(s), ptr)
            })
            .unwrap_or_else(|_| {
                let s = s.as_bytes();
                let idx = s
                    .iter()
                    .enumerate()
                    .find_map(|(idx, v)| if v == &0 { Some(idx) } else { None })
                    .unwrap();
                (None, s[..idx].as_ptr() as *const _)
            })
    }
}

impl Predictor {
    /// 执行模型预测，**需要在设置输入Tensor数据后调用**
    pub fn run(&self) -> bool {
        unsafe { PD_PredictorRun(self.ptr) }
    }
}

impl Clone for Predictor {
    fn clone(&self) -> Self {
        let ptr = unsafe { PD_PredictorClone(self.ptr) };
        Self { ptr }
    }
}

impl Drop for Predictor {
    fn drop(&mut self) {
        unsafe {
            PD_PredictorDestroy(self.ptr);
        }
    }
}
