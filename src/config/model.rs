use crate::call;
use crate::config::SetConfig;
use crate::ctypes::{PD_Config, PD_ConfigSetModel, PD_ConfigSetModelBuffer, PD_ConfigSetModelDir};
use crate::utils::to_c_str;

/// 预测模型
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(untagged))]
#[derive(Debug, Clone)]
pub enum Model {
    /// 设置模型文件路径，当需要从磁盘加载**非Combined**模型时使用
    ///
    /// **注意：** 非Combined 模型主要是为了兼容2.0之前版本的模型格式，从paddle 2.0开始，模型的默认保存格式是 Combined。
    Dir(String),
    /// 设置模型文件路径，当需要从磁盘加载**Combined**模型时使用。
    Path {
        /// Combined 模型文件所在路径
        model_file_path: String,
        /// Combined 模型参数文件所在路径
        params_file_path: String,
    },
    /// 从内存中加载预测模型
    Memory {
        /// 内存中模型结构数据
        model: Vec<u8>,
        /// 内存中模型参数数据
        params: Vec<u8>,
    },
}

impl Model {
    pub fn dir<S: ToString>(model_dir_path: S) -> Self {
        Self::Dir(model_dir_path.to_string())
    }

    pub fn path<S: ToString>(model_file_path: S, params_file_path: S) -> Self {
        Self::Path {
            model_file_path: model_file_path.to_string(),
            params_file_path: params_file_path.to_string(),
        }
    }
}

impl SetConfig for Model {
    fn set_to(self, config: *mut PD_Config) {
        match self {
            Model::Dir(dir) => {
                let (_p, ptr) = to_c_str(&dir);
                call! { PD_ConfigSetModelDir(config, ptr) };
            }
            Model::Path {
                model_file_path,
                params_file_path,
            } => {
                let (_m, model_path) = to_c_str(&model_file_path);
                let (_p, params_path) = to_c_str(&params_file_path);
                call! { PD_ConfigSetModel(config, model_path, params_path) };
            }
            Model::Memory { model, params } => {
                call! {
                    PD_ConfigSetModelBuffer(
                        config,
                        model.as_ptr() as *const _,
                        model.len(),
                        params.as_ptr() as *const _,
                        params.len()
                    )
                };
            }
        }
    }
}
