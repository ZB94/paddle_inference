//! [`crate::predictor::Predictor`]的构造器

pub mod lite_engine;
pub mod model;
pub mod setting;

use crate::call;
use crate::config::lite_engine::LiteEngine;
use crate::config::setting::{Cpu, Gpu, ONNXRuntime, Xpu};
use crate::ctypes::{
    PD_Config, PD_ConfigCreate, PD_ConfigDisableFCPadding, PD_ConfigDisableGlogInfo,
    PD_ConfigEnableMemoryOptim, PD_ConfigProfileEnabled, PD_ConfigSetOptimCacheDir,
    PD_ConfigSwitchIrDebug, PD_PredictorCreate,
};
use crate::predictor::Predictor;
use crate::utils::to_c_str;
use model::Model;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Config {
    /// 模型设置
    pub model: Model,
    /// CPU 配置
    pub cpu: Cpu,
    /// GPU 配置
    pub gpu: Option<Gpu>,
    /// XPU 配置
    pub xpu: Option<Xpu>,
    /// ONNXRuntime 设置
    pub onnx_runtime: Option<ONNXRuntime>,
    /// 启用 IR 优化, 默认打开
    pub ir_optimization: bool,
    /// 是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件, 默认关闭
    pub ir_debug: bool,
    /// 启用 Lite 子图
    pub lite: Option<LiteEngine>,
    /// 开启内存/显存复用，具体降低内存效果取决于模型结构
    pub memory_optimization: bool,
    /// 缓存路径
    ///
    /// **注意：** 如果当前使用的为 TensorRT INT8 且设置从内存中加载模型，则必须通过该方法来设置缓存路径。
    pub optimization_cache_dir: Option<String>,
    /// 禁用 FC Padding
    pub disable_fc_padding: bool,
    /// 打开 Profile，运行结束后会打印所有 OP 的耗时占比。
    pub profile: bool,
    /// 去除 Paddle Inference 运行中的 LOG
    pub disable_log: bool,
}

impl Config {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            cpu: Default::default(),
            gpu: None,
            xpu: None,
            onnx_runtime: None,
            ir_optimization: true,
            ir_debug: false,
            lite: None,
            memory_optimization: false,
            optimization_cache_dir: None,
            disable_fc_padding: false,
            profile: false,
            disable_log: false,
        }
    }

    /// 去除 Paddle Inference 运行中的 LOG
    pub fn disable_log_info(mut self) -> Self {
        self.disable_log = true;
        self
    }

    /// 打开 Profile，运行结束后会打印所有 OP 的耗时占比。
    pub fn enable_profile(mut self) -> Self {
        self.profile = true;
        self
    }

    /// 禁用 FC Padding
    pub fn disable_fc_padding(mut self) -> Self {
        self.disable_fc_padding = true;
        self
    }

    /// 设置缓存路径
    ///
    /// **注意：** 如果当前使用的为 TensorRT INT8 且设置从内存中加载模型，则必须通过该方法来设置缓存路径。
    pub fn set_optimization_cache_dir<S: ToString>(mut self, dir: S) -> Self {
        self.optimization_cache_dir = Some(dir.to_string());
        self
    }

    /// 开启内存/显存复用，具体降低内存效果取决于模型结构
    pub fn enable_memory_optimization(mut self) -> Self {
        self.memory_optimization = true;
        self
    }

    /// 启用 Lite 子图
    pub fn enable_lite_engine(mut self, lite_engine: LiteEngine) -> Self {
        self.lite = Some(lite_engine);
        self
    }

    /// 启用 IR 优化, 默认打开
    pub fn ir_optimization(mut self, enable: bool) -> Self {
        self.ir_optimization = enable;
        self
    }

    /// 是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件, 默认关闭
    pub fn ir_debug(mut self, debug: bool) -> Self {
        self.ir_debug = debug;
        self
    }

    pub fn cpu(mut self, cpu: Cpu) -> Self {
        self.cpu = cpu;
        self
    }

    pub fn gpu(mut self, gpu: Gpu) -> Self {
        self.gpu = Some(gpu);
        self
    }

    pub fn xpu(mut self, xpu: Xpu) -> Self {
        self.xpu = Some(xpu);
        self
    }

    pub fn onnx_runtime(mut self, onnx_runtime: ONNXRuntime) -> Self {
        self.onnx_runtime = Some(onnx_runtime);
        self
    }

    pub fn build(self) -> Predictor {
        let Self {
            model,
            cpu,
            gpu,
            xpu,
            onnx_runtime,
            ir_optimization,
            ir_debug,
            lite,
            memory_optimization,
            optimization_cache_dir,
            disable_fc_padding,
            profile,
            disable_log,
        } = self;

        let config = call! { PD_ConfigCreate() };

        model.set_to(config);

        cpu.set_to(config);
        if let Some(g) = gpu {
            g.set_to(config)
        }
        if let Some(x) = xpu {
            x.set_to(config)
        }
        if let Some(o) = onnx_runtime {
            o.set_to(config)
        }

        call! { PD_ConfigSwitchIrDebug(config, ir_optimization) };
        call! { PD_ConfigSwitchIrDebug(config, ir_debug) };

        if let Some(l) = lite {
            l.set_to(config)
        }

        call! { PD_ConfigEnableMemoryOptim(config, memory_optimization) };

        if let Some(s) = optimization_cache_dir {
            let (_s, cs) = to_c_str(&s);
            call! { PD_ConfigSetOptimCacheDir(config, cs) };
        }

        if disable_fc_padding {
            call! { PD_ConfigDisableFCPadding(config) };
        }

        if profile {
            call! { PD_ConfigProfileEnabled(config) };
        }

        if disable_log {
            call! { PD_ConfigDisableGlogInfo(config) };
        }

        let ptr = call! { PD_PredictorCreate(config) };
        Predictor::from_ptr(ptr)
    }
}

trait SetConfig {
    fn set_to(self, config: *mut PD_Config);
}
