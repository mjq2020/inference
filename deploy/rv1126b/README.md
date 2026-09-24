# RV1126B Workflow engine

RV1126B 轻量 Workflow 引擎维护在 [mjq2020/inference 的 rv1126b 分支](https://github.com/mjq2020/inference/tree/rv1126b)。

应用中心入口、manifest、图标、应用打包和生命周期测试已迁入
[recamera-pro-ext-api/apps/inference-rv1126b](https://github.com/Seeed-Studio/recamera-pro-ext-api/tree/main/apps/inference-rv1126b)。
请从 ext 的 `apps/inference-rv1126b/build.py` 构建应用；ext 的锁文件固定本仓完整提交号。
旧 `build_app.py --sdk-root <ext>` 入口会转交 ext 的构建脚本，并校验当前源码提交。

本仓继续维护：

- `inference/core/workflows/`：Workflow 执行引擎和节点。
- `inference/edge/`：RKNN、相机、RTSP、在线画布和设备服务适配。
- `deploy/rv1126b/build_wheel.py`：经裁剪的引擎、Supervision 和 NetworkX wheels。
- `requirements.txt`、`source_files.txt`、`download_wheels.py`、`audit_wheels.py`：固定依赖、源码白名单、离线依赖准备和审核。
- `MODEL_FORMAT.md`、`model.example.json`：RKNN 模型元数据契约。
- `tests/edge/`：引擎测试；应用专属测试随应用迁入 ext。

引擎不依赖设备上的 PyTorch、ONNX Runtime、训练模块或本地模型转换工具。
受应用中心管理的进程通过 Kit、FrameSource 和调度推理服务获取设备资源。
完整工作流说明见 [RV1126B Workflows](../../docs/rv1126b_workflows.md)。
应用监听地址、访问令牌、安装与升级说明以 ext 中的应用 README 为准。
