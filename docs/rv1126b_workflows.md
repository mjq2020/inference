> 应用入口、配置和安装包构建现由 [ext/apps/inference-rv1126b](https://github.com/Seeed-Studio/recamera-pro-ext-api/tree/main/apps/inference-rv1126b) 维护，默认局域网访问。本文描述引擎工作原理；历史应用构建命令请改用 ext 的构建入口。

# RV1126B 0.3.1 Workflow 使用与支持范围

设备发行版通过 `INFERENCE_RUNTIME_PROFILE=rv1126b` 选择 NumPy/RKNN 路径，仍使用
本项目的 `ExecutionEngine`、Workflow JSON、图编译器、批次关系和输出序列化。
0.2.0 以 Roboflow 原生画布为编排入口；无需安装 PyTorch、TorchVision、ONNX Runtime
或 `inference-models`。模型训练与模型转换在另一台机器完成。

## 原生画布、保存与设备运行

统一入口是设备 **App Center → Inference Workflows** 应用卡片。
在配置中设置应用访问口令和 `host=0.0.0.0`，选择“仅启动编辑服务”并启动应用，
然后从 **更多 → 编辑工作流** 复制完整临时连接地址、打开原生画布。
卡片使用设备管理登录换取临时连接授权，不再要求二次登录应用。
直接访问应用 `/` 或 `/build` 的旧连接入口仍保留，需要应用访问口令。
官方页面 `https://app.roboflow.com/workflows/local` 在浏览器中联网加载；
本地编排、Publish 保存和已适配节点的设备执行无需云端 API Key。

首次运行时，在官方画布顶部点击 **Locally → Other**，将设备入口显示的完整运行
地址粘贴到 **Running On** 的地址框，点击 **Connect**。保留地址中的
`/ui/runtime/…` 路径，并在浏览器提示时允许本地网络访问。该临时授权绑定设备登录
会话，12 小时、退出或应用重启后失效；重新登录后重新连接，不把长期设备口令放入 URL。

使用原来的节点、连线、变量、输入输出和测试面板。当前官方 `local` 页面中的
**Save** 草稿操作存在 `saveDraft` 前端错误，使用 **Publish** 保存到当前连接的设备。
以 `POST /build/api/{id}` 返回 201、刷新或重新打开后仍能读取为保存成功的依据。
设备的保存 API 不会触发摄像头启动，也不会切换已运行的实例。

持续推理统一在应用卡片配置：选择已保存 Workflow、设备摄像头或 RTSP、参数和帧率。
保存后启动应用，会执行所选流程；工作流为空时仅运行编辑服务。
配置表单的工作流列表在应用停止时也可读取。旧 `workflow_autostart` 配置不再控制应用中心启动。
流程必须恰好声明一个图像输入；多图像和批次可使用普通 Workflow API。

“查看结果”显示当前实例的真实图像输出和 JSON，RTSP 不会被替换为设备摄像头画面。
预览图像需在 Workflow 输出中选择 Box/Label Visualization 等节点的 `image`；
仅 JSON 输出的流程仍可持续处理视频。关闭结果窗口不会停止任务，点击“停止”释放资源。
旧 `/device` 自动跳转到卡片结果窗口，不再提供独立控制页。

启动时冻结定义和参数；Publish 不会改变当前实例，重启应用后才采用新定义。
所选任务成功处理首帧后才报告应用就绪，后续执行错误会反馈给应用管理器。
完整配置及持久化说明见[部署文档](../deploy/rv1126b/README.md)。

## 原生画布的 RTSP / Webcam 预览限制

2026-09-15 核对当前官方前端发现：本地无账户模式选择 RTSP / Webcam 并运行时，
可能显示 `Missing video preview configuration. Refresh the preview and try again.`。
该文案由官方页面在检查 `apiKey`、`serverURL`、`workflowConfigJSON` 和
`videoPreviewProps` 时直接生成；`videoPreviewProps` 的构造也要求云端 API Key。
缺少该 Key 时，WebRTC 组件不会挂载，不会向设备发送 `/initialise_webrtc_worker`。
因此这条报错本身不能证明 RTSP 地址、解码器或 RKNN 模型有问题。

官方本地入口接收 `serverUrl` 和 `csrf`，预览使用的 Key 则从 Roboflow 用户/工作空间
授权获取。当前无账户本地连接没有该 Key，也没有填写它的本地预览配置入口。
设备访问口令与设备配置中的 `roboflow_api_key` 均不会自动填入官方前端的这个状态；
不能以修改设备口令或反复刷新作为该情形的修复。真实云端登录可能改变 Key 的获取，
但尚未进行该路径的验收。

目前已验证的范围是原生画布编排、Publish 保存、单图运行，以及官方 SDK 的
ManualSource、MP4 和旧协议 WebRTC。**SDK 验收没有覆盖原生画布 RTSP / Webcam，
也没有证明用户的具体 RTSP 地址可拉流。** 0.3.1 的应用卡片已提供独立 RTSP 输入配置。
RTSP 后端入口使用原视频 API 的 `video_configuration.video_reference`，或 WebRTC
的 `rtsp_url`；原生画布内直接 Run 仍需要修复前端的本地授权兼容。应用卡片执行原画布保存的 Workflow JSON；这不等于修复官方画布内的预览按钮。

触发条件见[官方预览脚本](https://cdnassets.roboflow.com/_app/6064.be9ec8cffc3dbe30a1fc.js)，
本地入口见[官方本地画布脚本](https://cdnassets.roboflow.com/_app/5169.83dbac5a4107998554da.js)。
这些是当前线上版本的调查依据，不是项目可控制的前端接口契约。

同日用真实 Chromium 连接测试设备，打开已保存且含可视化输出的
`custom-workflow-2`，通过 Other 配置完整设备地址，选择 RTSP 并点击 Run，已复现
同一报错。点击后浏览器记录到的设备请求为 0，WebRTC 初始化为 0；匿名用户和工作空间
均为空。测试使用保留测试网段 URL，仅验证前端阻断，不作为实际 RTSP 拉流测试。
没有保存该测试输入、修改设备配置或启动推理任务。

## 节点和模型范围

当前目录有 **121 个节点版本条目**，包含同类节点的 v1/v2/v3，并非 121 种模型，也
不表示上游所有可选节点均已安装。实际类型、字段、别名和输出以
`GET /workflows/blocks/describe` 为准；源码目录由
[`catalog_rv1126b.py`](../inference/core/workflows/core_steps/catalog_rv1126b.py) 明确列出。

| 类别 | 当前目录中的代表能力 |
|---|---|
| 模型推理 | 原目标检测节点 v1/v2/v3，适配到本地 RKNN 模型管理器 |
| 图像与检测变换 | 绝对/相对/动态裁剪、切片、透视校正、合并/过滤/变换检测结果 |
| 跟踪与分析 | 原 Supervision ByteTracker v1/v2/v3、越线计数、区域停留、速度、轨迹分析 |
| 控制与格式 | ContinueIf、SwitchCase、RateLimiter、嵌套 Workflow、表达式、属性/字符串/JSON 处理 |
| 传统视觉 | 灰度、阈值、轮廓、模糊、形态学、运动检测、模板匹配、SIFT 等 |
| 可视化 | 框、标签、轨迹、区域、热图、Rich Label、Text Display 等原节点 |
| 融合和状态 | 检测共识/差异、缓冲、图像堆叠、帧延迟、CacheGet/CacheSet |

原 `roboflow_core/roboflow_object_detection_model@v1`、`@v2`、`@v3` 及各 manifest
声明的简写继续可用；`rv1126b/rknn_object_detection@v1` 也保留。`model_id` 指向已安装
的 RKNN 模型 ID，**不表示在线模型自动下载**。当前后端提供 YOLO 目标检测的
`yolo-decoded` 和 `yolo-dfl` 输出解码；模型元数据、输入预处理、输出布局和 RV1126B
运行时必须匹配。任意 RKNN 文件、分类/分割/关键点模型不能直接替换使用。

目录中存在分类标签、关键点可视化或 VLM 结果格式化节点，只说明它们可以处理对应
结构的数据，不意味着设备提供分类、关键点或 VLM 模型。任意 Python、自定义动态代码、
企业插件、其他重型模型及其 CPU 回退不启用；依赖尚未提供的节点也不会注册，例如
pandas CSV、scikit-image 对比度均衡、pycocotools mask visualization，以及另一个
`trackers` 包提供的跟踪器。可使用已注册的原 ByteTracker。

嵌套 Workflow 可内联，也可引用本地保存的 ID 或已授权的云端定义。解析前检查循环引用，
沿用原嵌套深度 4、展开数量 32 的约束。启动视频前解析并冻结子流程；之后修改子流程
只影响下一次启动。原查询语言、检测父图坐标和裁剪元数据传播继续由原引擎处理。

## HTTP 与官方 SDK

设备访问口令和 Roboflow API Key 分离。普通本地调用使用设备 `api_token`；支持
`Authorization: Bearer`、`X-Inference-Token`，以及官方 SDK 默认 JSON 请求体中的
`api_key`。访问云端 Workflow 时可以在应用配置中另填 `roboflow_api_key`，SDK 继续
使用设备口令。不会把设备口令发送到云端，云端 Key 也不代替设备认证。

| 接口 | 契约与用途 |
|---|---|
| `GET /workflows/execution_engine/versions` | 原执行引擎版本发现 |
| `GET` / `POST /workflows/blocks/describe` | 原节点描述，POST 可携带原描述请求字段 |
| `GET /workflows/definition/schema` | 原 Workflow 定义 schema |
| `POST /workflows/validate` | 请求体为 Workflow 定义本身，不包 `specification` |
| `POST /workflows/blocks/dynamic_outputs` | 请求体为单个节点 manifest，计算其动态输出；不启用自定义 Python |
| `POST /workflows/describe_interface` | 请求体包含 `specification`，返回输入/输出接口描述 |
| `POST /workflows/run` | 原 inline Workflow 请求与响应 |
| `POST /{workspace}/workflows/{id}` | 按 ID 执行；`workspace=local` 读取设备保存的定义 |
| `POST /{workspace}/workflows/{id}/describe_interface` | 按 ID 描述输入/输出接口 |
| `GET /build/api`，`GET` / `POST` / `DELETE /build/api/{id}` | 原生 Builder 列表、读取、保存、删除 |
| `GET /build/api/models` | 设备可用 RKNN 模型，标注原检测节点 v1/v2/v3 兼容类型 |

旧 `/infer/workflows` 及其按 ID 路由继续兼容。执行请求沿用原请求模型，传递
`debug`、`enable_profiling`、`is_preview`、`disable_sinks`、`excluded_fields` 等语义。
Profiling 还受原服务端 `ENABLE_WORKFLOWS_PROFILING` 开关控制；关闭时返回原空 trace。
`use_cache` 控制按 ID 获取的云端定义缓存，不代表缓存某次检测结果。内联定义不需要
云端定义缓存；`disable_sinks` 不会让未安装的外部 sink 节点变为可用。

云端 ID 支持有界内存/磁盘缓存及原 `workflow_version_id`；`use_cache=false` 强制
重新获取，不用离线快照替代。允许缓存且网络故障时可复用已有定义，云端认证失败不
以缓存绕过。本地保存记录没有云端发布版本，给本地 ID 指定版本会明确返回错误。

保存 API 保留原 `config` 字符串及未知画布字段。旧设备版本的顶层 `specification` /
`workflow` / `definition` 会在读取响应时适配原配置外层，保存前不破坏原文件。
这不承诺将旧自建画布的节点坐标转换成官方前端内部布局。

客户端安装正常官方 SDK，无需安装设备版：

```python
import json
import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

client = InferenceHTTPClient(
    api_url="http://设备IP:9001",
    api_key=os.environ["INFERENCE_DEVICE_TOKEN"],
)
client.configure(InferenceConfiguration(api_key_transport="header"))

with open("workflow.json", encoding="utf-8") as stream:
    specification = json.load(stream)
result = client.run_workflow(
    specification=specification,
    images={"image": "input.jpg"},
)

# 原生 Builder 已保存到此设备的 Workflow：
result = client.run_workflow(
    workspace_name="local",
    workflow_id="已保存的ID",
    images={"image": "input.jpg"},
)
```

图像输入名和参数名必须对应流程。客户端本地文件由 SDK 编码上传，服务端不读取调用者
指定的本地系统路径。HTTP 支持 base64、图像 data URI 和 HTTP(S) URL；URL 下载有
字节、像素、重定向及超时限制，不转发设备认证头或浏览器会话。Workflow 支持多个
图像输入、图像列表和 `WorkflowBatchInput`，所有解码图像累计受像素预算约束，NPU
逐张执行 batch=1。

## 最小裁剪与检测示例

向 `POST /workflows/run` 发送以下请求，替换图片 base64 即可运行原裁剪引擎：

```json
{
  "specification": {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [{
      "type": "AbsoluteStaticCrop",
      "name": "crop",
      "image": "$inputs.image",
      "x_center": 160,
      "y_center": 120,
      "width": 160,
      "height": 120
    }],
    "outputs": [{
      "type": "JsonField",
      "name": "crop",
      "selector": "$steps.crop.crops"
    }]
  },
  "inputs": {
    "image": {"type": "base64", "value": "<PNG_OR_JPEG_BASE64>"}
  }
}
```

HTTP 返回原 `outputs` 列表及 profiler/debug 字段；图像采用原 Workflow base64 图像
结构，官方 SDK 按其配置解码响应。只需要检测/计数时，可以只声明相应 JSON 输出。

检测步骤示例（`detector/1` 必须已安装）与其计数步骤：

```json
[
  {
    "type": "roboflow_core/roboflow_object_detection_model@v3",
    "name": "detect",
    "images": "$inputs.image",
    "model_id": "detector/1",
    "confidence": 0.4,
    "iou_threshold": 0.3,
    "max_detections": 100
  },
  {
    "type": "PropertyDefinition",
    "name": "count",
    "data": "$steps.detect.predictions",
    "operations": [{"type": "SequenceLength"}]
  }
]
```

输出计数的选择器是 `$steps.count.output`。这段为 `steps` 数组，需配合图像输入和
`JsonField` 输出构成完整 Workflow。远程 ONNX→RKNN 转换尚待用户提供 HTTP 接口；
当前 `/model-conversion` 会明确报告 `converter_unconfigured`，不会在设备运行转换工具。

## 视频与 WebRTC

原视频初始化协议使用 `VideoConfiguration`、`WorkflowConfiguration` 和
`MemorySinkConfiguration`。例如运行一个已保存的本地流程：

```json
{
  "video_configuration": {
    "type": "VideoConfiguration",
    "video_reference": "kit://camera",
    "max_fps": 3
  },
  "processing_configuration": {
    "type": "WorkflowConfiguration",
    "workspace_name": "local",
    "workflow_id": "已保存的ID",
    "image_input_name": "image",
    "video_metadata_input_name": "video_metadata",
    "workflows_parameters": {}
  },
  "sink_configuration": {
    "type": "MemorySinkConfiguration",
    "results_buffer_size": 2
  }
}
```

发送到 `POST /inference_pipelines/initialise`，从 `context.pipeline_id` 取得 ID。
使用原 `/{id}/status`、`/{id}/consume` GET 和 `/{id}/pause`、`/{id}/resume`、
`/{id}/terminate` POST，前缀均为 `/inference_pipelines`。消费响应包含
`outputs` 和 `frames_metadata`；无结果时返回空列表。旧设备版请求格式仍兼容。

视频输入包括 Kit 摄像头、支持的 RTSP/MJPEG URL；WebRTC 还接收客户端视频轨道和
原分块协议上传的视频文件。`POST /initialise_webrtc_worker` 及旧
`POST /inference_pipelines/initialise_webrtc` 提供真实 SDP/ICE 和数据通道处理，
不是仅模拟握手；新版与旧版消息格式各自保持。原画布/SDK 的实际可用性还取决于
浏览器本地网络权限、编解码器、ICE 网络连通性和已安装模型。

图像携带原 `VideoMetadata`，声明匹配名称的 `WorkflowVideoMetadata` 输入还会独立
收到视频 ID、帧号、时间戳和可用的帧率。可视化图像输出允许返回；WebRTC 预览按
`max_video_output_pixels` 缩小，模型输入和 JSON 检测坐标仍保持原分辨率。

同一应用同时保留一个活动视频会话；摄像头、WebRTC 及 HTTP 模型操作之间有互斥。
最近结果最多保留两条，超出后淘汰旧结果。暂停保留会话以便恢复；停止会等待借用帧
归还并释放模型/图实例。释放失败时显示错误、保留占用状态供重试，不假报释放成功。
关闭浏览器不会自动等同于停止设备摄像头；应明确调用 terminate 或设备页的停止。

## 资源预算与错误

当前默认请求上限 32 MiB、业务响应 8 MiB、最多 128 个展开后的步骤，配置范围见
部署文档。节点目录/schema/接口描述使用独立的 16 MiB 有界响应，保留完整 manifest
说明，不受旧设备 1 MiB 业务响应配置误伤。

每次执行仍有设备预算：最多 32 个裁剪、累计裁剪像素不超过配置输入像素上限的两倍，
最多 32 次模型调用；同一次执行的批次和嵌套裁剪共享预算。图像输出累计像素也有上限，
查询次数、中间 JSON 条目和深度有界。超限明确返回 `resource_budget` / HTTP 422，
不静默少返回检测结果。这些是当前实现的资源边界，不代表原 Workflow JSON 本身的限制。

HTTP 执行使用新图实例，避免 ContinueIf、跟踪、CacheGet/Set 等状态跨独立请求共享；
同一视频复用图实例，停止或淘汰时关闭节点持有的状态。执行步骤并发为 1，视频图实例
缓存默认最多 4 项。图像像素/请求体超限返回 413；原编译和输入验证错误通常为 400，
请求结构校验为 422；模型/NPU 或节点运行错误保留原错误上下文及对应状态码。

## 验证记录

开发机在没有 Torch、ONNX Runtime、`inference-models` 的设备测试环境中，已经覆盖
121 条目录的导入、原 Engine 的跟踪/越线计数/可视化与嵌套流程、缓存释放、批次和
坐标、HTTP 错误与选项、原配置格式，以及真实 Kit 配置映射和后台自动启动取消。
独立完整 SDK 环境使用未修改的官方 HTTP 客户端，经真实 Uvicorn 验证请求体/header/both
认证与 inline/local ID 六种组合。媒体测试实际交换现代/旧版 WebRTC 的视频和数据，
并验证文件上传、预览缩放和清理。

```sh
INFERENCE_RUNTIME_PROFILE=rv1126b python3 -m pytest -q \
  tests/edge/test_workflow_catalog.py \
  tests/edge/test_workflow_compatibility.py \
  tests/edge/test_video_compat.py \
  tests/edge/test_image_urls.py
```

软件模型管理器和模拟相机测试不计作真实 NPU 验收。另于 2026-09-15 在 RV1126B
完成以下真实设备验证，安装版本为 `0.2.0-b469c6bc93c19ab9`：

- **原生画布**：无需 Roboflow 账户，Publish 返回 201，刷新和重开保留
  `custom-workflow-2`（“RV1126B 原生检测与可视化”）；原 4 节点检测→计数→框→标签
  流程 Run 返回 200、计数为 2。
- **官方 HTTP SDK**：24 项通过。v1/v2/v3 检测节点分别完成请求体/header/both ×
  inline/local 的 18 种组合；类别/置信度参数、原 Filter、两图批次和本地嵌套通过。
  真实 1920×1080 图像返回两个 `dog` 检测，计数一致；框和标签相对同编码的原图改变
  69112 个像素，输出尺寸不变。缺少图像时原 SDK 收到 400 错误。仅本轮创建的四个
  测试草稿已删除，模型已释放。
- **设备摄像头**：`/device` 执行原 4 节点流程，Kit 输入 1280×720、处理上限 3 fps，
  累计 437 帧。帧数 381 时暂停并保持 2 秒不增长，恢复后增长至 437；停止后模型释放。
- **真实自动启动**：应用中心保存 `workflow_id`、`workflow_autostart=true` 及参数后
  重启，PID 1938→2821、generation 17；`/workflow-deployment` 返回目标流程 `running`。
  收到 23 帧时，同期服务记录 NPU 已完成 85 次调用、失败 0。冻结定义摘要为
  `1c5827ab7d635e5693734e98b7a5d179e832ef5e7391be8c8ccdd9209ca26ee6`。
- **原 WebRTC**：官方 SDK ManualSource 收到 4 份 JSON/3 帧视频预览，MP4 收到 4 份
  JSON/4 帧图像，旧协议收到 4 份 JSON/3 帧预览。使用真实 `yolo8n/1` RKNN；1080p
  输入的 720p 预览仍保留原检测坐标。达到目标 JSON 数后主动停止的实时用例少收到
  一帧视频，不等于 MP4 的逐帧图像路径丢帧；三项均确认会话和模型释放。

以上为功能和短时资源验证。HTTP 单图约 660–792 ms（首例约 1048 ms）、双图约
1279 ms，包含网络和序列化，不能当作纯 NPU 耗时。PSS 初始 82.18 MiB，打开原生目录后
约 270 MiB；直接重启只运行摄像头时应用为 216.13 MiB、共享 `inferenced` 为 37.24 MiB。
同一活动 WebRTC 会话的同步样本中，应用最大 468.095 MiB、共享服务 37.205 MiB，
合计约 505.30 MiB。共享服务不是应用的第二个进程；这些样本不是长时峰值或内存上界。

验收结束保留旧三个流程、新原生示例及原有口令/端口；恢复自动启动关闭、默认流程 ID
为空，HTTP 服务继续运行、摄像头停止。原始 HTTP SDK、自动启动、WebRTC 验收记录
和同步资源样本保存在本次工作区；完整版本摘要、包校验及大包上传的固件 nginx/JWT
配置要求见[部署文档](../deploy/rv1126b/README.md)。
