//! paddle inference C 接口

#![allow(rustdoc::bare_urls)]
#![allow(rustdoc::broken_intra_doc_links)]
#![allow(non_camel_case_types)]

#[link(name = "paddle_inference_c")]
extern "C" {}

pub type PD_Bool = bool;

/// Tensor 的数据精度, 默认值为[`DataType::Float32`]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(i32)]
pub enum DataType {
    Unknown = -1,
    Float32,
    Int32,
    Int64,
    Uint8,
}

impl Default for DataType {
    fn default() -> Self {
        Self::Float32
    }
}

/// 模型的运行精度, 默认值为[`PrecisionType::Float32`]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(i32)]
pub enum PrecisionType {
    Float32 = 0,
    Int8,
    Half,
}

impl Default for PrecisionType {
    fn default() -> Self {
        Self::Float32
    }
}

/// 目标设备硬件类型，用户可以根据应用场景选择硬件平台类型
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(i32)]
pub enum PlaceType {
    Unknown = -1,
    Cpu,
    Gpu,
    Xpu,
}

pub type PD_DataType = DataType;
pub type PD_PrecisionType = PrecisionType;
pub type PD_PlaceType = PlaceType;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PD_OneDimArrayInt32 {
    pub size: usize,
    pub data: *mut i32,
}
#[test]
fn bindgen_test_layout_PD_OneDimArrayInt32() {
    assert_eq!(
        ::std::mem::size_of::<PD_OneDimArrayInt32>(),
        16usize,
        concat!("Size of: ", stringify!(PD_OneDimArrayInt32))
    );
    assert_eq!(
        ::std::mem::align_of::<PD_OneDimArrayInt32>(),
        8usize,
        concat!("Alignment of ", stringify!(PD_OneDimArrayInt32))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_OneDimArrayInt32>())).size as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_OneDimArrayInt32),
            "::",
            stringify!(size)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_OneDimArrayInt32>())).data as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_OneDimArrayInt32),
            "::",
            stringify!(data)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PD_OneDimArraySize {
    pub size: usize,
    pub data: *mut usize,
}
#[test]
fn bindgen_test_layout_PD_OneDimArraySize() {
    assert_eq!(
        ::std::mem::size_of::<PD_OneDimArraySize>(),
        16usize,
        concat!("Size of: ", stringify!(PD_OneDimArraySize))
    );
    assert_eq!(
        ::std::mem::align_of::<PD_OneDimArraySize>(),
        8usize,
        concat!("Alignment of ", stringify!(PD_OneDimArraySize))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_OneDimArraySize>())).size as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_OneDimArraySize),
            "::",
            stringify!(size)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_OneDimArraySize>())).data as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_OneDimArraySize),
            "::",
            stringify!(data)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PD_OneDimArrayCstr {
    pub size: usize,
    pub data: *mut *mut ::std::os::raw::c_char,
}
#[test]
fn bindgen_test_layout_PD_OneDimArrayCstr() {
    assert_eq!(
        ::std::mem::size_of::<PD_OneDimArrayCstr>(),
        16usize,
        concat!("Size of: ", stringify!(PD_OneDimArrayCstr))
    );
    assert_eq!(
        ::std::mem::align_of::<PD_OneDimArrayCstr>(),
        8usize,
        concat!("Alignment of ", stringify!(PD_OneDimArrayCstr))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_OneDimArrayCstr>())).size as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_OneDimArrayCstr),
            "::",
            stringify!(size)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_OneDimArrayCstr>())).data as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_OneDimArrayCstr),
            "::",
            stringify!(data)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PD_Cstr {
    pub size: usize,
    pub data: *mut ::std::os::raw::c_char,
}
#[test]
fn bindgen_test_layout_PD_Cstr() {
    assert_eq!(
        ::std::mem::size_of::<PD_Cstr>(),
        16usize,
        concat!("Size of: ", stringify!(PD_Cstr))
    );
    assert_eq!(
        ::std::mem::align_of::<PD_Cstr>(),
        8usize,
        concat!("Alignment of ", stringify!(PD_Cstr))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_Cstr>())).size as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_Cstr),
            "::",
            stringify!(size)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_Cstr>())).data as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_Cstr),
            "::",
            stringify!(data)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PD_TwoDimArraySize {
    pub size: usize,
    pub data: *mut *mut PD_OneDimArraySize,
}
#[test]
fn bindgen_test_layout_PD_TwoDimArraySize() {
    assert_eq!(
        ::std::mem::size_of::<PD_TwoDimArraySize>(),
        16usize,
        concat!("Size of: ", stringify!(PD_TwoDimArraySize))
    );
    assert_eq!(
        ::std::mem::align_of::<PD_TwoDimArraySize>(),
        8usize,
        concat!("Alignment of ", stringify!(PD_TwoDimArraySize))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_TwoDimArraySize>())).size as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_TwoDimArraySize),
            "::",
            stringify!(size)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<PD_TwoDimArraySize>())).data as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(PD_TwoDimArraySize),
            "::",
            stringify!(data)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PD_Config {
    _unused: [u8; 0],
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Create a paddle config"]
    #[doc = ""]
    #[doc = " \\return new config."]
    #[doc = ""]
    pub fn PD_ConfigCreate() -> *mut PD_Config;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Destroy the paddle config"]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigDestroy(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set the combined model with two specific pathes for program and"]
    #[doc = " parameters."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] prog_file_path model file path of the combined model."]
    #[doc = " \\param[in] params_file_path params file path of the combined model."]
    #[doc = ""]
    pub fn PD_ConfigSetModel(
        pd_config: *mut PD_Config,
        prog_file_path: *const ::std::os::raw::c_char,
        params_file_path: *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set the model file path of a combined model."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] prog_file_path model file path."]
    #[doc = ""]
    pub fn PD_ConfigSetProgFile(
        pd_config: *mut PD_Config,
        prog_file_path: *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set the params file path of a combined model."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] params_file_path params file path."]
    #[doc = ""]
    pub fn PD_ConfigSetParamsFile(
        pd_config: *mut PD_Config,
        params_file_path: *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set the path of optimization cache directory."]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] opt_cache_dir the path of optimization cache directory."]
    #[doc = ""]
    pub fn PD_ConfigSetOptimCacheDir(
        pd_config: *mut PD_Config,
        opt_cache_dir: *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set the no-combined model dir path."]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] model_dir model dir path."]
    #[doc = ""]
    pub fn PD_ConfigSetModelDir(
        pd_config: *mut PD_Config,
        model_dir: *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the model directory path."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return The model directory path."]
    #[doc = ""]
    pub fn PD_ConfigGetModelDir(pd_config: *mut PD_Config) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the program file path."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return The program file path."]
    #[doc = ""]
    pub fn PD_ConfigGetProgFile(pd_config: *mut PD_Config) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the params file path."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return The params file path."]
    #[doc = ""]
    pub fn PD_ConfigGetParamsFile(pd_config: *mut PD_Config) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn off FC Padding."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigDisableFCPadding(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether fc padding is used."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether fc padding is used."]
    #[doc = ""]
    pub fn PD_ConfigUseFcPadding(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on GPU."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] memory_pool_init_size_mb initial size of the GPU memory pool in"]
    #[doc = " MB."]
    #[doc = " \\param[in] device_id device_id the GPU card to use."]
    #[doc = ""]
    pub fn PD_ConfigEnableUseGpu(
        pd_config: *mut PD_Config,
        memory_pool_init_size_mb: u64,
        device_id: i32,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn off GPU."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigDisableGpu(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the GPU is turned on."]
    #[doc = ""]
    #[doc = " \\brief Turn off GPU."]
    #[doc = " \\return Whether the GPU is turned on."]
    #[doc = ""]
    pub fn PD_ConfigUseGpu(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on ONNXRuntime."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableONNXRuntime(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn off ONNXRuntime."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigDisableONNXRuntime(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the ONNXRutnime is turned on."]
    #[doc = ""]
    #[doc = " \\return Whether the ONNXRuntime is turned on."]
    #[doc = ""]
    pub fn PD_ConfigONNXRuntimeEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on ONNXRuntime Optimization."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableORTOptimization(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on XPU."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param l3_workspace_size The size of the video memory allocated by the l3"]
    #[doc = "         cache, the maximum is 16M."]
    #[doc = " \\param locked Whether the allocated L3 cache can be locked. If false,"]
    #[doc = "       it means that the L3 cache is not locked, and the allocated L3"]
    #[doc = "       cache can be shared by multiple models, and multiple models"]
    #[doc = "       sharing the L3 cache will be executed sequentially on the card."]
    #[doc = " \\param autotune Whether to autotune the conv operator in the model. If"]
    #[doc = "       true, when the conv operator of a certain dimension is executed"]
    #[doc = "       for the first time, it will automatically search for a better"]
    #[doc = "       algorithm to improve the performance of subsequent conv operators"]
    #[doc = "       of the same dimension."]
    #[doc = " \\param autotune_file Specify the path of the autotune file. If"]
    #[doc = "       autotune_file is specified, the algorithm specified in the"]
    #[doc = "       file will be used and autotune will not be performed again."]
    #[doc = " \\param precision Calculation accuracy of multi_encoder"]
    #[doc = " \\param adaptive_seqlen Is the input of multi_encoder variable length"]
    #[doc = ""]
    pub fn PD_ConfigEnableXpu(
        pd_config: *mut PD_Config,
        l3_workspace_size: i32,
        locked: PD_Bool,
        autotune: PD_Bool,
        autotune_file: *const ::std::os::raw::c_char,
        precision: *const ::std::os::raw::c_char,
        adaptive_seqlen: PD_Bool,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on NPU."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] device_id device_id the NPU card to use."]
    #[doc = ""]
    pub fn PD_ConfigEnableNpu(pd_config: *mut PD_Config, device_id: i32);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the XPU is turned on."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether the XPU is turned on."]
    #[doc = ""]
    pub fn PD_ConfigUseXpu(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the NPU is turned on."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether the NPU is turned on."]
    #[doc = ""]
    pub fn PD_ConfigUseNpu(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the GPU device id."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return The GPU device id."]
    #[doc = ""]
    pub fn PD_ConfigGpuDeviceId(pd_config: *mut PD_Config) -> i32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the XPU device id."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return The XPU device id."]
    #[doc = ""]
    pub fn PD_ConfigXpuDeviceId(pd_config: *mut PD_Config) -> i32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the NPU device id."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return The NPU device id."]
    #[doc = ""]
    pub fn PD_ConfigNpuDeviceId(pd_config: *mut PD_Config) -> i32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the initial size in MB of the GPU memory pool."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return The initial size in MB of the GPU memory pool."]
    #[doc = ""]
    pub fn PD_ConfigMemoryPoolInitSizeMb(pd_config: *mut PD_Config) -> i32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the proportion of the initial memory pool size compared to the"]
    #[doc = " device."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return The proportion of the initial memory pool size."]
    #[doc = ""]
    pub fn PD_ConfigFractionOfGpuMemoryForPool(pd_config: *mut PD_Config) -> f32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on CUDNN."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableCudnn(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether to use CUDNN."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether to use CUDNN."]
    #[doc = ""]
    pub fn PD_ConfigCudnnEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Control whether to perform IR graph optimization."]
    #[doc = " If turned off, the AnalysisConfig will act just like a NativeConfig."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] x Whether the ir graph optimization is actived."]
    #[doc = ""]
    pub fn PD_ConfigSwitchIrOptim(pd_config: *mut PD_Config, x: PD_Bool);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the ir graph optimization is"]
    #[doc = " actived."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether to use ir graph optimization."]
    #[doc = ""]
    pub fn PD_ConfigIrOptim(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on the TensorRT engine."]
    #[doc = " The TensorRT engine will accelerate some subgraphes in the original Fluid"]
    #[doc = " computation graph. In some models such as resnet50, GoogleNet and so on,"]
    #[doc = " it gains significant performance acceleration."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] workspace_size The memory size(in byte) used for TensorRT"]
    #[doc = " workspace."]
    #[doc = " \\param[in] max_batch_size The maximum batch size of this prediction task,"]
    #[doc = " better set as small as possible for less performance loss."]
    #[doc = " \\param[in] min_subgrpah_size The minimum TensorRT subgraph size needed, if a"]
    #[doc = " subgraph is smaller than this, it will not be transferred to TensorRT"]
    #[doc = " engine."]
    #[doc = " \\param[in] precision The precision used in TensorRT."]
    #[doc = " \\param[in] use_static Serialize optimization information to disk for"]
    #[doc = " reusing."]
    #[doc = " \\param[in] use_calib_mode Use TRT int8 calibration(post training"]
    #[doc = " quantization)."]
    #[doc = ""]
    pub fn PD_ConfigEnableTensorRtEngine(
        pd_config: *mut PD_Config,
        workspace_size: i32,
        max_batch_size: i32,
        min_subgraph_size: i32,
        precision: PD_PrecisionType,
        use_static: PD_Bool,
        use_calib_mode: PD_Bool,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the TensorRT engine is used."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether the TensorRT engine is used."]
    #[doc = ""]
    pub fn PD_ConfigTensorRtEngineEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set min, max, opt shape for TensorRT Dynamic shape mode."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] tensor_num The number of the subgraph input."]
    #[doc = " \\param[in] tensor_name The name of every subgraph input."]
    #[doc = " \\param[in] shapes_num The shape size of every subgraph input."]
    #[doc = " \\param[in] min_shape The min input shape of every subgraph input."]
    #[doc = " \\param[in] max_shape The max input shape of every subgraph input."]
    #[doc = " \\param[in] optim_shape The opt input shape of every subgraph input."]
    #[doc = " \\param[in] disable_trt_plugin_fp16 Setting this parameter to true means that"]
    #[doc = " TRT plugin will not run fp16."]
    #[doc = ""]
    pub fn PD_ConfigSetTrtDynamicShapeInfo(
        pd_config: *mut PD_Config,
        tensor_num: usize,
        tensor_name: *mut *const ::std::os::raw::c_char,
        shapes_num: *mut usize,
        min_shape: *mut *mut i32,
        max_shape: *mut *mut i32,
        optim_shape: *mut *mut i32,
        disable_trt_plugin_fp16: PD_Bool,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the trt dynamic_shape is used."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigTensorRtDynamicShapeEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Enable tuned tensorrt dynamic shape."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] shape_range_info_path the path to shape_info file got in"]
    #[doc = " CollectShapeInfo mode."]
    #[doc = " \\param[in] allow_build_at_runtime allow build trt engine at runtime."]
    #[doc = ""]
    pub fn PD_ConfigEnableTunedTensorRtDynamicShape(
        pd_config: *mut PD_Config,
        shape_range_info_path: *const ::std::os::raw::c_char,
        allow_build_at_runtime: PD_Bool,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether to use tuned tensorrt dynamic"]
    #[doc = " shape."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigTunedTensorRtDynamicShape(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether to allow building trt engine at"]
    #[doc = " runtime."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigTrtAllowBuildAtRuntime(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Collect shape info of all tensors in compute graph."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] shape_range_info_path the path to save shape info."]
    #[doc = ""]
    pub fn PD_ConfigCollectShapeRangeInfo(
        pd_config: *mut PD_Config,
        shape_range_info_path: *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief the shape info path in CollectShapeInfo mode."]
    #[doc = " Attention, Please release the string manually."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigShapeRangeInfoPath(pd_config: *mut PD_Config) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether to collect shape info."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigShapeRangeInfoCollected(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Prevent ops running in Paddle-TRT"]
    #[doc = " NOTE: just experimental, not an official stable API, easy to be broken."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] ops_num ops number"]
    #[doc = " \\param[in] ops_name ops name"]
    #[doc = ""]
    pub fn PD_ConfigDisableTensorRtOPs(
        pd_config: *mut PD_Config,
        ops_num: usize,
        ops_name: *mut *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Replace some TensorRT plugins to TensorRT OSS("]
    #[doc = " https://github.com/NVIDIA/TensorRT), with which some models's inference"]
    #[doc = " may be more high-performance. Libnvinfer_plugin.so greater than"]
    #[doc = " V7.2.1 is needed."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableTensorRtOSS(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether to use the TensorRT OSS."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether to use the TensorRT OSS."]
    #[doc = ""]
    pub fn PD_ConfigTensorRtOssEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Enable TensorRT DLA"]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] dla_core ID of DLACore, which should be 0, 1,"]
    #[doc = "        ..., IBuilder.getNbDLACores() - 1"]
    #[doc = ""]
    pub fn PD_ConfigEnableTensorRtDla(pd_config: *mut PD_Config, dla_core: i32);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether to use the TensorRT DLA."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether to use the TensorRT DLA."]
    #[doc = ""]
    pub fn PD_ConfigTensorRtDlaEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on the usage of Lite sub-graph engine."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] precision Precion used in Lite sub-graph engine."]
    #[doc = " \\param[in] zero_copy whether use zero copy."]
    #[doc = " \\param[in] passes_filter_num The number of passes used in Lite sub-graph"]
    #[doc = " engine."]
    #[doc = " \\param[in] passes_filter The name of passes used in Lite sub-graph engine."]
    #[doc = " \\param[in] ops_filter_num The number of operators not supported by Lite."]
    #[doc = " \\param[in] ops_filter The name of operators not supported by Lite."]
    #[doc = ""]
    pub fn PD_ConfigEnableLiteEngine(
        pd_config: *mut PD_Config,
        precision: PD_PrecisionType,
        zero_copy: PD_Bool,
        passes_filter_num: usize,
        passes_filter: *mut *const ::std::os::raw::c_char,
        ops_filter_num: usize,
        ops_filter: *mut *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state indicating whether the Lite sub-graph engine is"]
    #[doc = " used."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether the Lite sub-graph engine is used."]
    #[doc = ""]
    pub fn PD_ConfigLiteEngineEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Control whether to debug IR graph analysis phase."]
    #[doc = " This will generate DOT files for visualizing the computation graph after"]
    #[doc = " each analysis pass applied."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] x whether to debug IR graph analysis phase."]
    #[doc = ""]
    pub fn PD_ConfigSwitchIrDebug(pd_config: *mut PD_Config, x: PD_Bool);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on MKLDNN."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableMKLDNN(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set the cache capacity of different input shapes for MKLDNN."]
    #[doc = " Default value 0 means not caching any shape."]
    #[doc = " Please see MKL-DNN Data Caching Design Document:"]
    #[doc = " https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md"]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] capacity The cache capacity."]
    #[doc = ""]
    pub fn PD_ConfigSetMkldnnCacheCapacity(pd_config: *mut PD_Config, capacity: i32);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether to use the MKLDNN."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether to use the MKLDNN."]
    #[doc = ""]
    pub fn PD_ConfigMkldnnEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set the number of cpu math library threads."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param cpu_math_library_num_threads The number of cpu math library"]
    #[doc = " threads."]
    #[doc = ""]
    pub fn PD_ConfigSetCpuMathLibraryNumThreads(
        pd_config: *mut PD_Config,
        cpu_math_library_num_threads: i32,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief An int state telling how many threads are used in the CPU math"]
    #[doc = " library."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return The number of threads used in the CPU math library."]
    #[doc = ""]
    pub fn PD_ConfigGetCpuMathLibraryNumThreads(pd_config: *mut PD_Config) -> i32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Specify the operator type list to use MKLDNN acceleration."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] ops_num The number of operator type list."]
    #[doc = " \\param[in] op_list The name of operator type list."]
    #[doc = ""]
    pub fn PD_ConfigSetMkldnnOp(
        pd_config: *mut PD_Config,
        ops_num: usize,
        op_list: *mut *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on MKLDNN quantization."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableMkldnnQuantizer(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the MKLDNN quantization is enabled."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether the MKLDNN quantization is enabled."]
    #[doc = ""]
    pub fn PD_ConfigMkldnnQuantizerEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on MKLDNN bfloat16."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableMkldnnBfloat16(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether to use the MKLDNN Bfloat16."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether to use the MKLDNN Bfloat16."]
    #[doc = ""]
    pub fn PD_ConfigMkldnnBfloat16Enabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = " \\brief Specify the operator type list to use Bfloat16 acceleration."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] ops_num The number of operator type list."]
    #[doc = " \\param[in] op_list The name of operator type list."]
    #[doc = ""]
    pub fn PD_ConfigSetBfloat16Op(
        pd_config: *mut PD_Config,
        ops_num: usize,
        op_list: *mut *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Enable the GPU multi-computing stream feature."]
    #[doc = " NOTE: The current behavior of this interface is to bind the computation"]
    #[doc = " stream to the thread, and this behavior may be changed in the future."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableGpuMultiStream(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the thread local CUDA stream is"]
    #[doc = " enabled."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether the thread local CUDA stream is enabled."]
    #[doc = ""]
    pub fn PD_ConfigThreadLocalStreamEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Specify the memory buffer of program and parameter."]
    #[doc = " Used when model and params are loaded directly from memory."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\param[in] prog_buffer The memory buffer of program."]
    #[doc = " \\param[in] prog_buffer_size The size of the model data."]
    #[doc = " \\param[in] params_buffer The memory buffer of the combined parameters file."]
    #[doc = " \\param[in] params_buffer_size The size of the combined parameters data."]
    #[doc = ""]
    pub fn PD_ConfigSetModelBuffer(
        pd_config: *mut PD_Config,
        prog_buffer: *const ::std::os::raw::c_char,
        prog_buffer_size: usize,
        params_buffer: *const ::std::os::raw::c_char,
        params_buffer_size: usize,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the model is set from the CPU"]
    #[doc = " memory."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether model and params are loaded directly from memory."]
    #[doc = ""]
    pub fn PD_ConfigModelFromMemory(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on memory optimize"]
    #[doc = " NOTE still in development."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableMemoryOptim(pd_config: *mut PD_Config, x: PD_Bool);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the memory optimization is"]
    #[doc = " activated."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether the memory optimization is activated."]
    #[doc = ""]
    pub fn PD_ConfigMemoryOptimEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Turn on profiling report."]
    #[doc = " If not turned on, no profiling report will be generated."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigEnableProfile(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the profiler is activated."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return bool Whether the profiler is activated."]
    #[doc = ""]
    pub fn PD_ConfigProfileEnabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Mute all logs in Paddle inference."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigDisableGlogInfo(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether logs in Paddle inference are muted."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether logs in Paddle inference are muted."]
    #[doc = ""]
    pub fn PD_ConfigGlogInfoDisabled(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set the Config to be invalid."]
    #[doc = " This is to ensure that an Config can only be used in one"]
    #[doc = " Predictor."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigSetInvalid(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief A boolean state telling whether the Config is valid."]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = " \\return Whether the Config is valid."]
    #[doc = ""]
    pub fn PD_ConfigIsValid(pd_config: *mut PD_Config) -> PD_Bool;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Partially release the memory"]
    #[doc = ""]
    #[doc = " \\param[in] pd_onfig config"]
    #[doc = ""]
    pub fn PD_ConfigPartiallyRelease(pd_config: *mut PD_Config);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Delete all passes that has a certain type 'pass'."]
    #[doc = ""]
    #[doc = " \\param[in] pass the certain pass type to be deleted."]
    #[doc = ""]
    pub fn PD_ConfigDeletePass(pd_config: *mut PD_Config, pass: *const ::std::os::raw::c_char);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief  Insert a pass to a specific position"]
    #[doc = ""]
    #[doc = " \\param[in] idx the position to insert."]
    #[doc = " \\param[in] pass the new pass."]
    #[doc = ""]
    pub fn PD_ConfigInsertPass(
        pd_config: *mut PD_Config,
        idx: usize,
        pass: *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Append a pass to the end of the passes"]
    #[doc = ""]
    #[doc = " \\param[in] pass the new pass."]
    #[doc = ""]
    pub fn PD_ConfigAppendPass(pd_config: *mut PD_Config, pass: *const ::std::os::raw::c_char);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get information of passes."]
    #[doc = ""]
    #[doc = " \\return Return list of the passes."]
    #[doc = ""]
    pub fn PD_ConfigAllPasses(pd_config: *mut PD_Config) -> *mut PD_OneDimArrayCstr;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get information of config."]
    #[doc = " Attention, Please release the string manually."]
    #[doc = ""]
    #[doc = " \\return Return config info."]
    #[doc = ""]
    pub fn PD_ConfigSummary(pd_config: *mut PD_Config) -> *mut PD_Cstr;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PD_Predictor {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PD_Tensor {
    _unused: [u8; 0],
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Create a new Predictor"]
    #[doc = ""]
    #[doc = " \\param[in] Config config"]
    #[doc = " \\return new predicor."]
    #[doc = ""]
    pub fn PD_PredictorCreate(pd_config: *mut PD_Config) -> *mut PD_Predictor;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Clone a new Predictor"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = " \\return new predictor."]
    #[doc = ""]
    pub fn PD_PredictorClone(pd_predictor: *mut PD_Predictor) -> *mut PD_Predictor;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the input names"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = " \\return input names"]
    #[doc = ""]
    pub fn PD_PredictorGetInputNames(pd_predictor: *mut PD_Predictor) -> *mut PD_OneDimArrayCstr;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the output names"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = " \\return output names"]
    #[doc = ""]
    pub fn PD_PredictorGetOutputNames(pd_predictor: *mut PD_Predictor) -> *mut PD_OneDimArrayCstr;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the input number"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = " \\return input number"]
    #[doc = ""]
    pub fn PD_PredictorGetInputNum(pd_predictor: *mut PD_Predictor) -> usize;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the output number"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = " \\return output number"]
    #[doc = ""]
    pub fn PD_PredictorGetOutputNum(pd_predictor: *mut PD_Predictor) -> usize;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the Input Tensor object"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = " \\param[in] name input name"]
    #[doc = " \\return input tensor"]
    #[doc = ""]
    pub fn PD_PredictorGetInputHandle(
        pd_predictor: *mut PD_Predictor,
        name: *const ::std::os::raw::c_char,
    ) -> *mut PD_Tensor;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the Output Tensor object"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = " \\param[in] name output name"]
    #[doc = " \\return output tensor"]
    #[doc = ""]
    pub fn PD_PredictorGetOutputHandle(
        pd_predictor: *mut PD_Predictor,
        name: *const ::std::os::raw::c_char,
    ) -> *mut PD_Tensor;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Run the prediction engine"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = " \\return Whether the function executed successfully"]
    #[doc = ""]
    pub fn PD_PredictorRun(pd_predictor: *mut PD_Predictor) -> PD_Bool;
}
extern "C" {
    #[doc = " \\brief Clear the intermediate tensors of the predictor"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = ""]
    pub fn PD_PredictorClearIntermediateTensor(pd_predictor: *mut PD_Predictor);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Release all tmp tensor to compress the size of the memory pool."]
    #[doc = " The memory pool is considered to be composed of a list of chunks, if"]
    #[doc = " the chunk is not occupied, it can be released."]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = " \\return Number of bytes released. It may be smaller than the actual"]
    #[doc = " released memory, because part of the memory is not managed by the"]
    #[doc = " MemoryPool."]
    #[doc = ""]
    pub fn PD_PredictorTryShrinkMemory(pd_predictor: *mut PD_Predictor) -> u64;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Destroy a predictor object"]
    #[doc = ""]
    #[doc = " \\param[in] pd_predictor predictor"]
    #[doc = ""]
    pub fn PD_PredictorDestroy(pd_predictor: *mut PD_Predictor);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get version info."]
    #[doc = ""]
    #[doc = " \\return version"]
    #[doc = ""]
    pub fn PD_GetVersion() -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Destroy the paddle tensor"]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor"]
    #[doc = ""]
    pub fn PD_TensorDestroy(pd_tensor: *mut PD_Tensor);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Reset the shape of the tensor."]
    #[doc = " Generally it's only used for the input tensor."]
    #[doc = " Reshape must be called before calling PD_TensorMutableData*() or"]
    #[doc = " PD_TensorCopyFromCpu*()"]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] shape_size The size of shape."]
    #[doc = " \\param[in] shape The shape to set."]
    #[doc = ""]
    pub fn PD_TensorReshape(pd_tensor: *mut PD_Tensor, shape_size: usize, shape: *mut i32);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer in CPU or GPU with 'float' data type."]
    #[doc = " Please Reshape the tensor first before call this."]
    #[doc = " It's usually used to get input data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] place The place of the tensor."]
    #[doc = " \\return Memory pointer of pd_tensor"]
    #[doc = ""]
    pub fn PD_TensorMutableDataFloat(pd_tensor: *mut PD_Tensor, place: PD_PlaceType) -> *mut f32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer in CPU or GPU with 'int64_t' data type."]
    #[doc = " Please Reshape the tensor first before call this."]
    #[doc = " It's usually used to get input data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] place The place of the tensor."]
    #[doc = " \\return Memory pointer of pd_tensor"]
    #[doc = ""]
    pub fn PD_TensorMutableDataInt64(pd_tensor: *mut PD_Tensor, place: PD_PlaceType) -> *mut i64;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer in CPU or GPU with 'int32_t' data type."]
    #[doc = " Please Reshape the tensor first before call this."]
    #[doc = " It's usually used to get input data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] place The place of the tensor."]
    #[doc = " \\return Memory pointer of pd_tensor"]
    #[doc = ""]
    pub fn PD_TensorMutableDataInt32(pd_tensor: *mut PD_Tensor, place: PD_PlaceType) -> *mut i32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer in CPU or GPU with 'uint8_t' data type."]
    #[doc = " Please Reshape the tensor first before call this."]
    #[doc = " It's usually used to get input data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] place The place of the tensor."]
    #[doc = " \\return Memory pointer of pd_tensor"]
    #[doc = ""]
    pub fn PD_TensorMutableDataUint8(pd_tensor: *mut PD_Tensor, place: PD_PlaceType) -> *mut u8;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer in CPU or GPU with 'int8_t' data type."]
    #[doc = " Please Reshape the tensor first before call this."]
    #[doc = " It's usually used to get input data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] place The place of the tensor."]
    #[doc = " \\return Memory pointer of pd_tensor"]
    #[doc = ""]
    pub fn PD_TensorMutableDataInt8(pd_tensor: *mut PD_Tensor, place: PD_PlaceType) -> *mut i8;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer directly."]
    #[doc = " It's usually used to get the output data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] place To get the device type of the tensor."]
    #[doc = " \\param[out] size To get the data size of the tensor."]
    #[doc = " \\return The tensor data buffer pointer."]
    #[doc = ""]
    pub fn PD_TensorDataFloat(
        pd_tensor: *mut PD_Tensor,
        place: *mut PD_PlaceType,
        size: *mut i32,
    ) -> *mut f32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer directly."]
    #[doc = " It's usually used to get the output data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] place To get the device type of the tensor."]
    #[doc = " \\param[out] size To get the data size of the tensor."]
    #[doc = " \\return The tensor data buffer pointer."]
    #[doc = ""]
    pub fn PD_TensorDataInt64(
        pd_tensor: *mut PD_Tensor,
        place: *mut PD_PlaceType,
        size: *mut i32,
    ) -> *mut i64;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer directly."]
    #[doc = " It's usually used to get the output data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] place To get the device type of the tensor."]
    #[doc = " \\param[out] size To get the data size of the tensor."]
    #[doc = " \\return The tensor data buffer pointer."]
    #[doc = ""]
    pub fn PD_TensorDataInt32(
        pd_tensor: *mut PD_Tensor,
        place: *mut PD_PlaceType,
        size: *mut i32,
    ) -> *mut i32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer directly."]
    #[doc = " It's usually used to get the output data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] place To get the device type of the tensor."]
    #[doc = " \\param[out] size To get the data size of the tensor."]
    #[doc = " \\return The tensor data buffer pointer."]
    #[doc = ""]
    pub fn PD_TensorDataUint8(
        pd_tensor: *mut PD_Tensor,
        place: *mut PD_PlaceType,
        size: *mut i32,
    ) -> *mut u8;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the memory pointer directly."]
    #[doc = " It's usually used to get the output data pointer."]
    #[doc = ""]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] place To get the device type of the tensor."]
    #[doc = " \\param[out] size To get the data size of the tensor."]
    #[doc = " \\return The tensor data buffer pointer."]
    #[doc = ""]
    pub fn PD_TensorDataInt8(
        pd_tensor: *mut PD_Tensor,
        place: *mut PD_PlaceType,
        size: *mut i32,
    ) -> *mut i8;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the host memory to tensor data."]
    #[doc = " It's usually used to set the input tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] data The pointer of the data, from which the tensor will copy."]
    #[doc = ""]
    pub fn PD_TensorCopyFromCpuFloat(pd_tensor: *mut PD_Tensor, data: *const f32);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the host memory to tensor data."]
    #[doc = " It's usually used to set the input tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] data The pointer of the data, from which the tensor will copy."]
    #[doc = ""]
    pub fn PD_TensorCopyFromCpuInt64(pd_tensor: *mut PD_Tensor, data: *const i64);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the host memory to tensor data."]
    #[doc = " It's usually used to set the input tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] data The pointer of the data, from which the tensor will copy."]
    #[doc = ""]
    pub fn PD_TensorCopyFromCpuInt32(pd_tensor: *mut PD_Tensor, data: *const i32);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the host memory to tensor data."]
    #[doc = " It's usually used to set the input tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] data The pointer of the data, from which the tensor will copy."]
    #[doc = ""]
    pub fn PD_TensorCopyFromCpuUint8(pd_tensor: *mut PD_Tensor, data: *const u8);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the host memory to tensor data."]
    #[doc = " It's usually used to set the input tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] data The pointer of the data, from which the tensor will copy."]
    #[doc = ""]
    pub fn PD_TensorCopyFromCpuInt8(pd_tensor: *mut PD_Tensor, data: *const i8);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the tensor data to the host memory."]
    #[doc = " It's usually used to get the output tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] data The tensor will copy the data to the address."]
    #[doc = ""]
    pub fn PD_TensorCopyToCpuFloat(pd_tensor: *mut PD_Tensor, data: *mut f32);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the tensor data to the host memory."]
    #[doc = " It's usually used to get the output tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] data The tensor will copy the data to the address."]
    #[doc = ""]
    pub fn PD_TensorCopyToCpuInt64(pd_tensor: *mut PD_Tensor, data: *mut i64);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the tensor data to the host memory."]
    #[doc = " It's usually used to get the output tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] data The tensor will copy the data to the address."]
    #[doc = ""]
    pub fn PD_TensorCopyToCpuInt32(pd_tensor: *mut PD_Tensor, data: *mut i32);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the tensor data to the host memory."]
    #[doc = " It's usually used to get the output tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] data The tensor will copy the data to the address."]
    #[doc = ""]
    pub fn PD_TensorCopyToCpuUint8(pd_tensor: *mut PD_Tensor, data: *mut u8);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Copy the tensor data to the host memory."]
    #[doc = " It's usually used to get the output tensor data."]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[out] data The tensor will copy the data to the address."]
    #[doc = ""]
    pub fn PD_TensorCopyToCpuInt8(pd_tensor: *mut PD_Tensor, data: *mut i8);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the tensor shape"]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\return The tensor shape."]
    #[doc = ""]
    pub fn PD_TensorGetShape(pd_tensor: *mut PD_Tensor) -> *mut PD_OneDimArrayInt32;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Set the tensor lod information"]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\param[in] lod lod information."]
    #[doc = ""]
    pub fn PD_TensorSetLod(pd_tensor: *mut PD_Tensor, lod: *mut PD_TwoDimArraySize);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the tensor lod information"]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\return the lod information."]
    #[doc = ""]
    pub fn PD_TensorGetLod(pd_tensor: *mut PD_Tensor) -> *mut PD_TwoDimArraySize;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the tensor name"]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\return the tensor name."]
    #[doc = ""]
    pub fn PD_TensorGetName(pd_tensor: *mut PD_Tensor) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Get the tensor data type"]
    #[doc = " \\param[in] pd_tensor tensor."]
    #[doc = " \\return the tensor data type."]
    #[doc = ""]
    pub fn PD_TensorGetDataType(pd_tensor: *mut PD_Tensor) -> PD_DataType;
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Destroy the PD_OneDimArrayInt32 object pointed to by the pointer."]
    #[doc = ""]
    #[doc = " \\param[in] array pointer to the PD_OneDimArrayInt32 object."]
    #[doc = ""]
    pub fn PD_OneDimArrayInt32Destroy(array: *mut PD_OneDimArrayInt32);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Destroy the PD_OneDimArrayCstr object pointed to by the pointer."]
    #[doc = ""]
    #[doc = " \\param[in] array pointer to the PD_OneDimArrayCstr object."]
    #[doc = ""]
    pub fn PD_OneDimArrayCstrDestroy(array: *mut PD_OneDimArrayCstr);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Destroy the PD_OneDimArraySize object pointed to by the pointer."]
    #[doc = ""]
    #[doc = " \\param[in] array pointer to the PD_OneDimArraySize object."]
    #[doc = ""]
    pub fn PD_OneDimArraySizeDestroy(array: *mut PD_OneDimArraySize);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Destroy the PD_TwoDimArraySize object pointed to by the pointer."]
    #[doc = ""]
    #[doc = " \\param[in] array pointer to the PD_TwoDimArraySize object."]
    #[doc = ""]
    pub fn PD_TwoDimArraySizeDestroy(array: *mut PD_TwoDimArraySize);
}
extern "C" {
    #[doc = ""]
    #[doc = " \\brief Destroy the PD_Cstr object pointed to by the pointer."]
    #[doc = " NOTE: if input string is empty, the return PD_Cstr's size is"]
    #[doc = " 0 and data is NULL."]
    #[doc = ""]
    #[doc = " \\param[in] cstr pointer to the PD_Cstr object."]
    #[doc = ""]
    pub fn PD_CstrDestroy(cstr: *mut PD_Cstr);
}
