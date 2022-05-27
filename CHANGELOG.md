# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**注意：** `0.2.0`及其之前的版本有文档和字段说明错误，最好不要使用，避免歧义。

## [Unreleased]

### Added

- 添加`serde` feature。启用时`config`中的所有结构及`common::{DataType, PlaceType, PrecisionType}`可以被序列化和反序列化

### Changed

- 为`config`中的所有结构实现`Clone`
- 公开`config::Config`的所有属性
- `config::Config::set_optimization_cache_dir`参数类型修改为实现`ToString`特质的结构

## 0.3.0

### Fixed

- 修复`config::setting::Xpu`的文档说明错误，将`l3_workspace_size_mb`重命名为`l3_workspace_size`

## ~~0.2.0~~

### Changed
- 添加`libloading`依赖，动态库加载方式修改为使用`libloading`动态加载
