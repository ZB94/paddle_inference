# Paddle Inference

本库是对百度飞浆推理库C接口的封装，详细说明请参考[官方文档](https://paddleinference.paddlepaddle.org.cn/api_reference/c_api_index.html)

## 使用说明

1. 编译前请先[下载或编译预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html)
2. 编译时请确保`paddle_inference_c`库（Windows下为`paddle_inference_c.lib`, Linux下为`libpaddle_inference_c.a`）处于能被链接器搜索的位置
（Windows MSVC下设置环境变量`LIB`/Linux下设置环境变量`LD_LIBRARY_PATH`并指向库所在**目录**）

## 使用示例
```rust
use paddle_inference::config::model::Model;
use paddle_inference::config::setting::Cpu;
use paddle_inference::Predictor;

let predictor = Predictor::builder(Model::path(
        "模型文件路径",
        "模型参数文件路径",
    ))
    // 使用 CPU 识别
    .cpu(Cpu {
        threads: Some(std::thread::available_parallelism().unwrap().get() as i32),
        mkldnn: None,
    })
    // 设置缓存陌路
    .set_optimization_cache_dir("caches".to_string())
    // 创建 Predictor
    .build();


let names = predictor.input_names();
println!("输入名称列表长度: {}", names.len());

// 获取和设置输入数据
let input = predictor.input(&names.get(0).unwrap());
input.reshape(&[1, 3, 100, 100]);
input.copy_from_f32(&[0.0; 3 * 100 * 100]);

// 执行
println!("run: {}", predictor.run());

let names = predictor.output_names();
println!("output names len: {}", names.len());

let output = predictor.output(&names.get(0).unwrap());
println!("output type: {:?}", output.data_type());
println!("output shape: {:?}", output.shape());
```
