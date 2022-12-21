## 使用方法

**注意：**

1. 运行之前需要先安装opencv，并配置好相关的环境变量。详见[opencv](https://crates.io/crates/opencv)
2. 仅测试过中文检测模型，使用其他语言模型可能调整。（主要为处理文本检测模型结果）

```
Usage: ocr [OPTIONS] <DET_MODEL_PATH> <DET_PARAMS_PATH> <REC_MODEL_PATH> <REC_PARAMS_PATH> <REC_LABEL_PATH> <IMAGE_PATH>

Arguments:
  <DET_MODEL_PATH>   文本检测模型路径
  <DET_PARAMS_PATH>  文本检测模型参数路径
  <REC_MODEL_PATH>   文本识别模型路径
  <REC_PARAMS_PATH>  文本识别模型参数路径
  <REC_LABEL_PATH>   文本识别标签路径
  <IMAGE_PATH>       要识别的图片路径

Options:
      --det-cache-dir <DET_CACHE_DIR>                          模型缓存目录 [default: out/det/]
      --det-result-dir <DET_RESULT_DIR>                        文件检测结果保存目录 [default: out/det_result]
      --rec-cahce-dir <REC_CAHCE_DIR>                          [default: out/rec/]
      --gpu                                                    是否使用GPU识别
      --cudnn                                                  使用启用cudnn
      --gpu-memory-pool-init-size <GPU_MEMORY_POOL_INIT_SIZE>  GPU内存池的初始化大小。单位为mb [default: 1024]
      --gpu-device-id <GPU_DEVICE_ID>                          [default: 0]
      --cpu-threads <CPU_THREADS>                              cpu线程数。小于或等于0时为系统线程数 [default: 0]
```
