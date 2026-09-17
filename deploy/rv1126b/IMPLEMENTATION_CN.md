# RV1126B 轻量应用实现

本目录提供独立的 `inference-rv1126b` 应用发行版，目标固件为
RV1126B、AArch64、CPython 3.11、RKNN Runtime 2.3.2。仓库默认完整运行模式继续保留；
设备发行包使用显式源码清单及独立依赖，不安装训练、PyTorch、TorchVision、
ONNX Runtime、`inference-models`、Transformers、模型导出或 RKNN 转换工具。

## 已实现

- HTTP 服务使用一个进程；RKNN 请求在该进程内通过平台 `RemoteRknnSession` 发给
  `inferenced`。模型按需加载，最多一个会话，切换与卸载显式释放。
- 原 Workflow JSON、编译器、执行器及坐标关系继续使用，注册 10 类 NumPy/RKNN
  节点。详见 [节点目录与请求示例](../../docs/rv1126b_workflows.md)。
- 单路按需相机流水线，默认最高 10 FPS；最近两条结果供 HTTP 消费。
  暂停/运行视频期间拒绝 HTTP 推理和模型变更，停止后可重新调用。
- 输入以 base64 PNG/JPEG 上传，解码前检查像素上限。模型输入为 RGB uint8，
  图像 API 与 Workflow 的内部数组沿用 BGR uint8。
- 包含严格模型 metadata、文件 SHA-256 校验、YOLO 已解码输出和 YOLO DFL 后处理。
  示例模型 `yolo8n/1` 对应已有六输出、类别 logits 的 YOLOv8n RKNN。
- 应用中心 manifest v2、按发行版隔离的离线依赖、Kit 生命周期入口、安装导入探针、
  包内容/BOM/hash/ARM ABI/依赖闭包校验及可复现构建。

## 内存边界

| 项目 | 首版设置 |
| --- | --- |
| 活跃模型 / 相机流水线 | 1 / 1 |
| Workflow 步骤并发 | 1 |
| HTTP 在途请求 | 最多 2，超限返回 429 |
| HTTP 请求体 | 默认 8 MiB |
| 每条结果 JSON | 默认 1 MiB |
| 输入图像 | 默认总计不超过 4096×2160 像素 |
| Workflow 步数 | 默认最多 32 |
| 图编译缓存 / 语法实体缓存 | 4 / 2 |
| 相机结果缓存 | 最近 2 条；消费后清除 |
| 累计裁剪 | 最多 32 次，累计像素不超过输入像素预算的两倍 |

视频逐帧处理，无独立的 64 帧应用队列。HTTP 输入、工作流中间图像与分支还需遵守
运行时预算；超限明确拒绝。manifest 中的 memory_mb 是平台准入额度，不能视作
操作系统强制内存隔离。NPU 实际执行位于共享 `inferenced`，整机验收应同时测量
应用 PSS、该服务的增量、驱动/DMA及 MemAvailable。

视频 Workflow 的最终输出只允许 JSON 结果，禁止将图像编码后存进结果缓存；
裁剪图像仍可在内部传给检测节点。停止/结束视频后释放相机、工作流实例和模型会话。

## 转换服务接入

设备直接使用已安装的 `.rknn` 模型。`inference/edge/conversion.py` 定义了
`ConversionRequest`、`ConversionResult` 和可注入的 `Converter` 协议。
当前尚无外部 HTTP 协议，`POST /model-conversion` 返回 503，
`GET /model-conversion` 报告未配置；设备不做 ONNX→RKNN 转换。

后续提供接口后，需要落实上传/下载方式、鉴权、同步或异步任务、超时/重试、
目标 `rv1126b`、量化校准、转换工具版本及输入/输出 metadata。转换结果应同时带
模型文件、哈希、类别表和张量/后处理契约。详见 [模型格式](MODEL_FORMAT.md)。

模型 metadata 不能授予 NPU 权限。新模型仍须进入应用中心认可的 artifact 与
发行版流程，由平台校验安装路径和哈希，之后才可加载。

## 启动与调用

开发机上可使用专用依赖环境启动 HTTP 与无模型 Workflow：

```sh
python inference_edge.py --model-root /path/to/model-store
```

默认监听 `127.0.0.1:9001`。设备安装后可通过 SSH 转发访问：

```sh
ssh -L 9001:127.0.0.1:9001 root@192.168.66.80
curl http://127.0.0.1:9001/healthz
curl http://127.0.0.1:9001/capabilities
curl http://127.0.0.1:9001/model/registry
```

应用中心启动使用 `app.py`，HTTP 和推理线程共享平台授权的主 PID。
直接用 SSH 启动脚本只适合无模型验证，无法替代 appmgr 的 NPU/相机授权。
启动成功表示 HTTP 服务就绪；模型和相机在第一次请求时加载。

| 接口 | 用途 |
| --- | --- |
| `GET /healthz`、`GET /capabilities` | 健康与支持能力 |
| `GET /model/registry` | 已安装模型及加载状态 |
| `POST /model/add`、`/model/remove`、`/model/clear` | 模型会话管理 |
| `POST /infer/object_detection` | 单张图像 RKNN 检测 |
| `POST /workflows/blocks/describe`、`/workflows/run` | 原 Workflow schema 与执行 |
| `POST /inference_pipelines/initialise` | 使用 `model_id` 或 `specification` 启动相机任务 |
| `POST /inference_pipelines/pause`、`/resume`、`/terminate` | 使用 `pipeline_id` 管理相机任务 |
| `POST /inference_pipelines/consume` | 取走最近结果 |
| `GET /model-conversion` | 转换服务配置状态 |

HTTP 对外监听需要配置 `INFERENCE_EDGE_API_TOKEN`，请求带 `Authorization: Bearer …`。
该 token 与兼容字段 `api_key` 不同。首版默认使用本机监听和 SSH 转发；
固件尚未提供通用应用 HTTP 路由，manifest 的逻辑端点不会自动生成 Web 入口。
相机结果通过上述 HTTP 接口返回；平台 Web 叠加层、MQTT、录像事件和可视化编辑器
尚未接入。原完整 HTTP SDK 的所有接口也不属于本发行版兼容范围。

## 构建和安装

先按 [依赖构建说明](README.md) 生成 ARM64 wheelhouse，再整理模型目录：

```text
model-store/
  yolo8n/1/
    model.json                 # 参考 model.example.json
    yolo8n_rawhead_int8.rknn    # 与 metadata SHA256 一致
```

```sh
python deploy/rv1126b/build_app.py \
  --sdk-root /path/to/recamera-pro-ext-api \
  --wheelhouse /path/to/wheelhouse \
  --model-root /path/to/model-store \
  --out /path/to/packages
```

构建输出为 `inference-rv1126b-0.1.2-arm64.tar.gz` 和摘要文件。构建不签名、不安装、
不自动启动。设备的正式安装策略要求有效签名；本地 Web 未签名安装例外要求用户
明确确认，并且不会自动启动。构建器保留该策略。

0.1.2 入口适配当前固件的 Kit 启动方式：私有环境依赖优先于系统旧版包；必要时
在同一 PID 内重新初始化解释器，保留应用中心身份和 READY 通知。应用只使用
HTTP 结果接口，因此入口为本应用选择官方 stdout sink，避免额外绑定默认 8124
端口。两项适配都在应用自身进程内完成，不修改固件 SDK 或系统 Python 包。

## 验证

在专用开发环境执行：

```sh
INFERENCE_RUNTIME_PROFILE=rv1126b python -m pytest tests/edge -q
python deploy/rv1126b/smoke_wheel.py /path/to/wheelhouse
PYTHONPATH=. python deploy/rv1126b/measure_runtime.py --iterations 100 --out memory.json
```

`measure_runtime.py` 启动真实 HTTP 服务，反复执行原裁剪 Workflow，读取
`/proc/self/smaps_rollup` 的 RSS/PSS、`/proc/self/status` 的峰值和线程数，并记录
重型框架的导入尝试。它不加载 NPU 模型或读取相机；该结果不能作为 NPU 总内存。
设备上可加 `--kit-app /path/to/app.py`，将实际 Kit 应用启动和清理也纳入测量。

设备验收还包括正式安装后的 NPU 精度、同图后处理对比、相机断流/停流、启动停止、
同负载性能与 PSS、24 小时稳定性、升级和回滚。实施测量与包摘要保存在 SDK 的
`artifacts/inference-rv1126b-implementation-20260914/`。
