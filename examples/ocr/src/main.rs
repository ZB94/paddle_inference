use opencv::{imgcodecs::imwrite, prelude::*};
use std::{error::Error, path::PathBuf, time::Instant};

use clap::Parser;
use ocr::{detect, recognize};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use paddle_inference::{
    config::{
        model::Model,
        setting::{Cpu, Gpu},
    },
    Predictor,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = Parser::parse();
    println!("{args:#?}");

    std::fs::create_dir_all(&args.det_result_dir)?;

    let det = args.det_model();
    let rec = args.rec_model();

    println!("已加载模型");

    let labels = std::fs::read_to_string(&args.rec_label_path)?
        .lines()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let image = imread(&args.image_path, IMREAD_COLOR)?;

    let t = Instant::now();
    let mut roi_list = detect(&det, &image)?;
    roi_list.sort_by_key(|r| r.y);

    for (idx, (line, roi)) in recognize(&rec, &image, &roi_list, 8, &labels)?
        .into_iter()
        .zip(roi_list.iter())
        .enumerate()
    {
        let path = args.det_result_dir.join(format!("{idx:03}.jpg"));

        imwrite(
            path.to_str().unwrap(),
            &Mat::roi(&image, *roi)?,
            &Default::default(),
        )?;

        println!("{idx:03} {line} {roi:?}");
    }

    println!("{:?}", t.elapsed());

    Ok(())
}

/// Paddle Inference Ocr
#[derive(Debug, Parser)]
#[command(author, version)]
pub struct Args {
    /// 文本检测模型路径
    pub det_model_path: String,
    /// 文本检测模型参数路径
    pub det_params_path: String,
    /// 模型缓存目录
    #[arg(long, default_value = "out/det/")]
    pub det_cache_dir: String,
    /// 文件检测结果保存目录
    #[arg(long, default_value = "out/det_result")]
    pub det_result_dir: PathBuf,

    /// 文本识别模型路径
    pub rec_model_path: String,
    /// 文本识别模型参数路径
    pub rec_params_path: String,
    #[arg(long, default_value = "out/rec/")]
    pub rec_cahce_dir: String,
    /// 文本识别标签路径
    pub rec_label_path: String,

    /// 要识别的图片路径
    pub image_path: String,

    /// 是否使用GPU识别
    #[arg(long)]
    pub gpu: bool,
    /// 使用启用cudnn
    #[arg(long)]
    pub cudnn: bool,
    /// GPU内存池的初始化大小。单位为mb
    #[arg(long, default_value_t = 1024)]
    pub gpu_memory_pool_init_size: u64,
    #[arg(long, default_value_t = 0)]
    pub gpu_device_id: i32,

    /// cpu线程数。小于或等于0时为系统线程数
    #[arg(long, default_value_t = 0)]
    pub cpu_threads: i32,
}

impl Args {
    fn gpu_config(&self) -> Option<Gpu> {
        self.gpu.then_some(Gpu {
            memory_pool_init_size_mb: self.gpu_memory_pool_init_size,
            device_id: self.gpu_device_id,
            enable_multi_stream: false,
            enable_cudnn: self.cudnn,
            enable_tensor_rt: None,
        })
    }

    fn cpu_config(&self) -> Cpu {
        Cpu {
            threads: (self.cpu_threads > 0)
                .then_some(self.cpu_threads)
                .or_else(|| Some(std::thread::available_parallelism().unwrap().get() as i32)),
            mkldnn: None,
        }
    }

    fn model(&self, model: Model, cache_dir: &str) -> Predictor {
        let mut p = Predictor::builder(model)
            .cpu(self.cpu_config())
            .enable_memory_optimization()
            .disable_log_info()
            .set_optimization_cache_dir(cache_dir);

        if let Some(g) = self.gpu_config() {
            p = p.gpu(g);
        }

        p.build()
    }

    pub fn det_model(&self) -> Predictor {
        self.model(
            Model::path(&self.det_model_path, &self.det_params_path),
            &self.det_cache_dir,
        )
    }

    pub fn rec_model(&self) -> Predictor {
        self.model(
            Model::path(&self.rec_model_path, &self.rec_params_path),
            &self.rec_cahce_dir,
        )
    }
}
