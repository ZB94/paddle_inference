use opencv::{imgcodecs::imwrite, prelude::*};
use std::{error::Error, time::Instant};

use ocr::{detect, recognize};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use paddle_inference::config::{model::Model, setting::Gpu};

fn main() -> Result<(), Box<dyn Error>> {
    let threads = std::thread::available_parallelism()?.get() as i32;

    let rec = paddle_inference::Predictor::builder(Model::path(
        "examples/ocr/resource/ch_ppocr_server_v2.0_rec_infer/inference.pdmodel",
        "examples/ocr/resource/ch_ppocr_server_v2.0_rec_infer/inference.pdiparams",
    ))
    .gpu(Gpu {
        memory_pool_init_size_mb: 1024,
        device_id: 0,
        enable_multi_stream: false,
        enable_cudnn: true,
        enable_tensor_rt: None,
    })
    .cpu(paddle_inference::config::setting::Cpu {
        threads: Some(threads),
        mkldnn: None,
    })
    .enable_memory_optimization()
    .set_optimization_cache_dir("target/rec")
    .disable_log_info()
    .build();

    let det = paddle_inference::Predictor::builder(Model::path(
        "examples/ocr/resource/ch_ppocr_server_v2.0_det_infer/inference.pdmodel",
        "examples/ocr/resource/ch_ppocr_server_v2.0_det_infer/inference.pdiparams",
    ))
    .gpu(Gpu {
        memory_pool_init_size_mb: 1024,
        device_id: 0,
        enable_multi_stream: false,
        enable_cudnn: true,
        enable_tensor_rt: None,
    })
    .cpu(paddle_inference::config::setting::Cpu {
        threads: Some(threads),
        mkldnn: None,
    })
    .enable_memory_optimization()
    .set_optimization_cache_dir("target/det")
    .disable_log_info()
    .build();

    let image = imread("examples/ocr/resource/test.jpg", IMREAD_COLOR)?;

    let t = Instant::now();
    let mut roi_list = detect(&det, &image)?;
    roi_list.sort_by_key(|r| r.y);

    for (idx, (line, roi)) in recognize(&rec, &image, &roi_list, 8)?
        .into_iter()
        .zip(roi_list.iter())
        .enumerate()
    {
        imwrite(
            &format!("out/{idx:03}.jpg"),
            &Mat::roi(&image, *roi)?,
            &Default::default(),
        )?;

        println!("{idx:03} {line} {roi:?}");
    }

    println!("{:?}", t.elapsed());

    Ok(())
}
