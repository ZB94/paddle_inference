use crate::common::PrecisionType;
use crate::config::SetConfig;
use crate::ctypes::{PD_Config, PD_ConfigEnableLiteEngine};
use crate::utils::to_c_str;

/// Lite 子图
#[derive(Debug)]
pub struct LiteEngine {
    /// Lite 子图的运行精度
    pub precision: PrecisionType,
    /// 启用 zero_copy，lite 子图与 paddle inference 之间共享数据
    pub zero_copy: bool,
    /// lite 子图的 pass 名称列表
    pub passes_filter: Vec<String>,
    /// 不使用 lite 子图运行的 op 名称列表
    pub ops_filter: Vec<String>,
}

impl SetConfig for LiteEngine {
    fn set_to(self, config: *mut PD_Config) {
        let LiteEngine {
            precision,
            zero_copy,
            passes_filter,
            ops_filter,
        } = self;
        let (_p, mut passes_filter_ptr) = passes_filter
            .iter()
            .map(|s| to_c_str(s))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let passes_filter_num = passes_filter_ptr.len();

        let (_o, mut ops_filter_ptr) = ops_filter
            .iter()
            .map(|s| to_c_str(s))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let ops_filter_num = ops_filter_ptr.len();

        unsafe {
            PD_ConfigEnableLiteEngine(
                config,
                precision,
                zero_copy,
                passes_filter_num,
                passes_filter_ptr.as_mut_ptr(),
                ops_filter_num,
                ops_filter_ptr.as_mut_ptr(),
            );
        }
    }
}
