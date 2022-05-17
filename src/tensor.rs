use crate::common::{DataType, OneDimArrayInt32, PlaceType, TwoDimArraySize};
use crate::ctypes::{
    PD_Tensor, PD_TensorCopyFromCpuFloat, PD_TensorCopyFromCpuInt32, PD_TensorCopyFromCpuInt64,
    PD_TensorCopyFromCpuInt8, PD_TensorCopyFromCpuUint8, PD_TensorCopyToCpuFloat,
    PD_TensorCopyToCpuInt32, PD_TensorCopyToCpuInt64, PD_TensorCopyToCpuInt8,
    PD_TensorCopyToCpuUint8, PD_TensorDataFloat, PD_TensorDataInt32, PD_TensorDataInt64,
    PD_TensorDataInt8, PD_TensorDataUint8, PD_TensorDestroy, PD_TensorGetDataType, PD_TensorGetLod,
    PD_TensorGetName, PD_TensorGetShape, PD_TensorMutableDataFloat, PD_TensorMutableDataInt32,
    PD_TensorMutableDataInt64, PD_TensorMutableDataInt8, PD_TensorMutableDataUint8,
    PD_TensorReshape, PD_TensorSetLod,
};
use std::borrow::Cow;
use std::ffi::CStr;

/// Tensor 是 Paddle Inference 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、
/// 数据、LoD 信息等。
pub struct Tensor {
    ptr: *mut PD_Tensor,
}

impl Tensor {
    pub fn from_ptr(ptr: *mut PD_Tensor) -> Self {
        Self { ptr }
    }
}

impl Tensor {
    /// 设置维度信息
    pub fn reshape(&self, shape: &[i32]) {
        unsafe {
            PD_TensorReshape(self.ptr, shape.len(), shape.as_ptr() as *mut _);
        }
    }

    /// 获取维度信息
    pub fn shape(&self) -> Vec<i32> {
        let ptr = unsafe { PD_TensorGetShape(self.ptr) };
        OneDimArrayInt32::from_ptr(ptr).to_vec()
    }

    pub fn data_type(&self) -> DataType {
        unsafe { PD_TensorGetDataType(self.ptr) }
    }

    pub fn name(&self) -> Cow<str> {
        let ptr = unsafe { PD_TensorGetName(self.ptr) };
        unsafe { CStr::from_ptr(ptr).to_string_lossy() }
    }
}

impl Tensor {
    pub fn copy_from_f32(&self, data: &[f32]) {
        unsafe {
            PD_TensorCopyFromCpuFloat(self.ptr, data.as_ptr());
        }
    }

    pub fn copy_from_i64(&self, data: &[i64]) {
        unsafe {
            PD_TensorCopyFromCpuInt64(self.ptr, data.as_ptr());
        }
    }

    pub fn copy_from_i32(&self, data: &[i32]) {
        unsafe {
            PD_TensorCopyFromCpuInt32(self.ptr, data.as_ptr());
        }
    }

    pub fn copy_from_u8(&self, data: &[u8]) {
        unsafe {
            PD_TensorCopyFromCpuUint8(self.ptr, data.as_ptr());
        }
    }

    pub fn copy_from_i8(&self, data: &[i8]) {
        unsafe {
            PD_TensorCopyFromCpuInt8(self.ptr, data.as_ptr());
        }
    }
}

impl Tensor {
    #[inline]
    fn size(&self) -> usize {
        self.shape().into_iter().fold(1usize, |s, i| s * i as usize)
    }

    fn check_data_type(&self, ty: DataType) -> bool {
        let dt = self.data_type();
        dt != DataType::Unknown && dt == ty
    }

    fn check(&self, size: usize, ty: DataType) -> bool {
        size >= self.size() && self.check_data_type(ty)
    }
}

impl Tensor {
    /// 从 Tensor 中获取数据，返回是否获取成功
    ///
    /// 如果出现以下情况则获取失败
    /// - 输入类型和[`Self::data_type`]不匹配
    /// - 输入数据大小小于[`Self::shape`]结果之积
    pub fn copy_to_f32(&self, data: &mut [f32]) -> bool {
        if self.check(data.len(), DataType::Float32) {
            unsafe {
                PD_TensorCopyToCpuFloat(self.ptr, data.as_mut_ptr());
            }
            true
        } else {
            false
        }
    }

    /// 从 Tensor 中获取数据，返回是否获取成功
    ///
    /// 如果出现以下情况则获取失败
    /// - 输入类型和[`Self::data_type`]不匹配
    /// - 输入数据大小小于[`Self::shape`]结果之积
    pub fn copy_to_i64(&self, data: &mut [i64]) -> bool {
        if self.check(data.len(), DataType::Int64) {
            unsafe {
                PD_TensorCopyToCpuInt64(self.ptr, data.as_mut_ptr());
            }
            true
        } else {
            false
        }
    }

    /// 从 Tensor 中获取数据，返回是否获取成功
    ///
    /// 如果出现以下情况则获取失败
    /// - 输入类型和[`Self::data_type`]不匹配
    /// - 输入数据大小小于[`Self::shape`]结果之积
    pub fn copy_to_i32(&self, data: &mut [i32]) -> bool {
        if self.check(data.len(), DataType::Int32) {
            unsafe {
                PD_TensorCopyToCpuInt32(self.ptr, data.as_mut_ptr());
            }
            true
        } else {
            false
        }
    }

    /// 从 Tensor 中获取数据，返回是否获取成功
    ///
    /// 如果出现以下情况则获取失败
    /// - 输入类型和[`Self::data_type`]不匹配
    /// - 输入数据大小小于[`Self::shape`]结果之积
    pub fn copy_to_u8(&self, data: &mut [u8]) -> bool {
        if self.check(data.len(), DataType::Uint8) {
            unsafe {
                PD_TensorCopyToCpuUint8(self.ptr, data.as_mut_ptr());
            }
            true
        } else {
            false
        }
    }

    /// 从 Tensor 中获取数据，返回是否获取成功
    ///
    /// 如果出现以下情况则获取失败
    /// - 输入类型和[`Self::data_type`]不匹配
    /// - 输入数据大小小于[`Self::shape`]结果之积
    pub fn copy_to_i8(&self, data: &mut [i8]) -> bool {
        if self.check(data.len(), DataType::Uint8) {
            unsafe {
                PD_TensorCopyToCpuInt8(self.ptr, data.as_mut_ptr());
            }
            true
        } else {
            false
        }
    }
}

impl Tensor {
    /// 获取 Tensor 底层数据，用于设置输入数据。
    ///
    /// **需要先调用[`Self::reshape`]**
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_mut_slice_f32(&self, place_type: PlaceType) -> Option<&mut [f32]> {
        self.check_data_type(DataType::Float32).then(|| {
            let ptr = unsafe { PD_TensorMutableDataFloat(self.ptr, place_type) };
            unsafe { std::slice::from_raw_parts_mut(ptr, self.size()) }
        })
    }

    /// 获取 Tensor 底层数据，用于设置输入数据。
    ///
    /// **需要先调用[`Self::reshape`]**
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_mut_slice_i64(&self, place_type: PlaceType) -> Option<&mut [i64]> {
        self.check_data_type(DataType::Int64).then(|| {
            let ptr = unsafe { PD_TensorMutableDataInt64(self.ptr, place_type) };
            unsafe { std::slice::from_raw_parts_mut(ptr, self.size()) }
        })
    }

    /// 获取 Tensor 底层数据，用于设置输入数据。
    ///
    /// **需要先调用[`Self::reshape`]**
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_mut_slice_i32(&self, place_type: PlaceType) -> Option<&mut [i32]> {
        self.check_data_type(DataType::Int32).then(|| {
            let ptr = unsafe { PD_TensorMutableDataInt32(self.ptr, place_type) };
            unsafe { std::slice::from_raw_parts_mut(ptr, self.size()) }
        })
    }

    /// 获取 Tensor 底层数据，用于设置输入数据。
    ///
    /// **需要先调用[`Self::reshape`]**
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_mut_slice_u8(&self, place_type: PlaceType) -> Option<&mut [u8]> {
        self.check_data_type(DataType::Uint8).then(|| {
            let ptr = unsafe { PD_TensorMutableDataUint8(self.ptr, place_type) };
            unsafe { std::slice::from_raw_parts_mut(ptr, self.size()) }
        })
    }

    /// 获取 Tensor 底层数据，用于设置输入数据。
    ///
    /// **需要先调用[`Self::reshape`]**
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_mut_slice_i8(&self, place_type: PlaceType) -> Option<&mut [i8]> {
        self.check_data_type(DataType::Uint8).then(|| {
            let ptr = unsafe { PD_TensorMutableDataInt8(self.ptr, place_type) };
            unsafe { std::slice::from_raw_parts_mut(ptr, self.size()) }
        })
    }
}

impl Tensor {
    /// 获取 Tensor 底层数据，用于读取输出数据。
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_slice_f32(&self) -> Option<(PlaceType, &[f32])> {
        self.check_data_type(DataType::Float32).then(|| {
            let mut place_type = PlaceType::Unknown;
            let mut size = 0;
            let ptr = unsafe { PD_TensorDataFloat(self.ptr, &mut place_type, &mut size) };
            let s = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
            (place_type, s)
        })
    }

    /// 获取 Tensor 底层数据，用于读取输出数据。
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_slice_i64(&self) -> Option<(PlaceType, &[i64])> {
        self.check_data_type(DataType::Int64).then(|| {
            let mut place_type = PlaceType::Unknown;
            let mut size = 0;
            let ptr = unsafe { PD_TensorDataInt64(self.ptr, &mut place_type, &mut size) };
            let s = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
            (place_type, s)
        })
    }

    /// 获取 Tensor 底层数据，用于读取输出数据。
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_slice_i32(&self) -> Option<(PlaceType, &[i32])> {
        self.check_data_type(DataType::Int32).then(|| {
            let mut place_type = PlaceType::Unknown;
            let mut size = 0;
            let ptr = unsafe { PD_TensorDataInt32(self.ptr, &mut place_type, &mut size) };
            let s = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
            (place_type, s)
        })
    }

    /// 获取 Tensor 底层数据，用于读取输出数据。
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_slice_u8(&self) -> Option<(PlaceType, &[u8])> {
        self.check_data_type(DataType::Uint8).then(|| {
            let mut place_type = PlaceType::Unknown;
            let mut size = 0;
            let ptr = unsafe { PD_TensorDataUint8(self.ptr, &mut place_type, &mut size) };
            let s = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
            (place_type, s)
        })
    }

    /// 获取 Tensor 底层数据，用于读取输出数据。
    ///
    /// 如果底层数据类型([`DataType`])不对应则返回`None`
    pub fn as_slice_i8(&self) -> Option<(PlaceType, &[i8])> {
        self.check_data_type(DataType::Uint8).then(|| {
            let mut place_type = PlaceType::Unknown;
            let mut size = 0;
            let ptr = unsafe { PD_TensorDataInt8(self.ptr, &mut place_type, &mut size) };
            let s = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
            (place_type, s)
        })
    }
}

impl Tensor {
    pub fn set_lod(&self, lod: TwoDimArraySize) {
        unsafe {
            PD_TensorSetLod(self.ptr, lod.ptr);
        }
    }

    pub fn lod(&self) -> TwoDimArraySize {
        let ptr = unsafe { PD_TensorGetLod(self.ptr) };
        TwoDimArraySize::from_ptr(ptr)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            PD_TensorDestroy(self.ptr);
        }
    }
}
