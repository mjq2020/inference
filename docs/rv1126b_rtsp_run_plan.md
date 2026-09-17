# 原生 Workflow 画布 RTSP Run 修复方案

日期：2026-09-15。本文是实施方案，未修改或升级设备程序。

## 交付目标

在原生画布编排流程，选择 RTSP、填写视频源及其认证信息，点击原 Run 按钮，由
RV1126B 拉流、执行原 Workflow 和 RKNN 推理，原测试面板显示结果。模型与流程均为
本地资源时，不要求 Roboflow 云端 API Key。

RTSP 是运行时视频源，每帧绑定到 Workflow 声明的图像输入。沿用原 Workflow JSON、
节点、输入选择器与跨帧状态；视频源设置不改写模型节点，也不新增另一套工作流格式。
测试执行当前画布定义，不能悄悄执行上次保存的旧版本。

## 已知事实与实施前提

上轮真实 Chromium 连接设备后，已在含图像输出的原流程上复现相同报错；点击后设备
请求和 WebRTC 初始化请求均为 0。官方前端的启动条件及 videoPreviewProps 构造依赖
云端 effectiveApiKey，本地设备会话没有填入该状态。详见
[调查报告](/home/dq/github/RV1126B_Linux_IPC_SDK/artifacts/inference-rv1126b-rtsp-preview-20260915/REPORT_CN.md)。

本仓库的 `inference/core/interfaces/http/builder/editor.html` 和
`inference/edge/static/builder.js` 是加载画布的包装层；完整画布从官方域名加载。
本次仓库核对未找到该视频面板的可构建前端源码。包装页不能直接修改跨域页面组件。
因此，**完全保留官方原画布 Run 且移除本地模式云 Key 依赖，必须具备可交付的前端修改
渠道**：官方发布修复，或提供可以维护、构建和部署的 Builder 前端版本。不能把更改
`BUILDER_ORIGIN` 本身当作已经获得完整可用的前端。

官方文档描述 Detached 无需账户和 API Key，但本次实际线上版本的视频入口存在缺口。
不能由 Inference 仓库开源推断完整在线 Builder 已包含在本仓或可直接打包分发。

## 实施路径

| 路径 | 原画布内 RTSP Run | 云端账户 / Key | 实施条件与定位 |
| --- | --- | --- | --- |
| 修复 Detached 前端 | 最终验收要求 | 本地资源不需要 | 官方修复或可维护的 Builder 前端；推荐的最终方案 |
| Cloud connected 画布连接本设备 | 官方支持的连接方式，实际 RTSP 闭环待测 | 需要真实账户与工作空间授权 | 可作为过渡；推理服务器显式选本设备，不能默认使用 Serverless |

如果前端修改渠道暂时无法落实，Detached 正式修复存在明确外部依赖。Cloud connected
需要用户自己的真实账户才能验证，不能承诺当前匿名 local 页面只要登录就必然可用。
其定义保存可能进入云端工作空间，须与设备本地保存区别说明；可导入同一 Workflow
定义进行测试，设备长期部署使用明确的本地定义。

增加独立设备页、SDK 脚本或浏览器注入补丁都不能作为原画布原 Run 已兼容的验收证据。
当前已存 SDK 视频链路通过记录，但尚无用户实际 RTSP 源的完整验收记录。

## Detached 前端需要改动的内容

1. 将视频运行授权区分为设备会话与云端授权。运行目标为已认证本地设备时，使用
   有效设备会话 / 临时运行地址；运行目标为 Roboflow 云端时，保持真实云 Key 授权。
   不以填入任意字符串伪装有效云 Key，也不关闭设备认证。
2. 同时修改 Run 就绪判断、videoPreviewProps 构造、WebRTC connector 初始化以及
   心跳 / 停止请求的认证处理。只删报错判断不足以启动整个通路。
3. 使用现有设备临时授权连接，所有接口保留完整 `/ui/runtime/{grant}` 前缀；只向
   已选设备发送设备认证。长期设备口令留在设备登录流程，禁止将其用作云端 API Key。
4. 运行当前画布 Workflow 定义，传递真实图像输入名、业务参数、视频元数据输入名、
   输出选择以及原 disable_sinks 语义。显示图像时要求选取实际存在的图像输出；数据
   输出与图像显示分开校验，不将无图像预览混同为无法执行推理。
5. 清楚显示连接设备、打开视频源、运行流程和输出结果各阶段错误，避免统一显示
   “Missing video preview configuration”。授权过期应重新连接；媒体错误应保留可
   定位原因，但不在日志、错误信息或分享链接中泄露 RTSP 密码。

## 设备后端改动与复用范围

已有 `/initialise_webrtc_worker` 和视频处理通路，不另造独立推理服务。连接路径为：

```text
原画布 Run ── 设备授权 + RTSP 地址 + 当前 Workflow + SDP ──> RV1126B
RTSP 摄像头 ── 视频 ──> 设备解码 ──> 原 Workflow ──> RKNN / NPU
原画布输出面板 <── WebRTC 画面 + JSON 结果 ──────────────────┘
```

沿用原字段 `rtsp_url`、`workflow_configuration.workflow_specification`、
`image_input_name`、`workflows_parameters`、`stream_output`、`data_output` 和
`webrtc_offer`。本地资源执行无需请求云端 Key。只有显式引用云端资源的流程才走相应
云端凭证解析，并在部署前提示依赖。

对应代码：

| 文件 | 工作 |
| --- | --- |
| `inference/edge/static/builder.js` | 对接最终可交付前端，传递本地运行目标和授权；自身无法修复跨域画布组件 |
| `inference/edge/browser.py` | 复用设备会话和临时授权，核实新前端的请求范围、过期及跨域行为 |
| `inference/edge/webrtc.py` | 以真实前端抓包核对初始化、SDP、结果通道、心跳和结束协议 |
| `inference/edge/streaming.py` | 核对当前定义、参数、输入名和结果选择的标准协议语义 |
| `inference/edge/video.py` | 验证实际 RTSP 的认证、解码、超时、停止和断流释放 |
| `tests/edge/` | 有意义的本地授权、会话生命周期和视频输出回归 |

已发现原完整服务有 `/webrtc/session/heartbeat/end`，当前 edge 的结束接口为
`/webrtc/session/end`。应按实际客户端需要补兼容并测试；它不是当前初始化前报错的
原因，不能单靠增加这一接口宣称问题解决。

默认复用现有 RTSP TCP / PyAV 解码，先完成真实链路；RKNN 负责模型计算，视频解码
和 WebRTC 编码的 CPU / 内存成本需另计。硬件解码优化按固件正式媒体接口能力另行
验证，不承诺仅使用 NPU 就能消除全部 CPU 和内存消耗。

## RV1126B 资源与生命周期

- 默认一个活动视频流程，复用引擎与模型会话。连续帧必须保留跟踪、计数等节点状态，
  不能改为每帧创建独立 HTTP 工作流。
- 默认处理上限 3 FPS，预览上限 1280×720，队列仅保留最新帧。缩放预览不改变原工作流
  图像、检测坐标或图像输出的业务语义。实际内存以完整 RTSP 链路测量为准。
- 原 Run 用于交互测试；Stop、连接断开、授权失效后在约定超时内释放媒体和执行会话。
  浏览器关闭不应残留无人管理的测试任务。长期无人值守任务使用独立部署生命周期。
- 保持训练、PyTorch、ONNX Runtime 和转换工具不进入设备应用；远程模型转换仍是
  模型准备阶段的独立功能。

## 实施顺序和验收

1. **落实前端修改渠道。** 在真实 Builder 上证明本地设备授权可进入 WebRTC 初始化，
   云端分支授权保持正确。若只有临时补丁或独立页面，只记录为技术验证。
2. **完成实际 RTSP 闭环。** 使用设备可访问的真实视频源，从原画布 Run 触发，获得
   JSON、可视化图像以及设备 NPU 调用证据；同时核对请求确实发向设备。
3. **完成失败与功能回归。** 检查错误密码、不可达地址、断流、Stop / 重开、授权过期、
   刷新 / 关闭页面、当前未保存定义、非默认输入名、参数、可视化、跟踪和计数。
4. **长时和资源测试。** 相同流程 / 分辨率 / 帧率下至少运行 30 分钟、重复启停 20 次，
   记录应用和共享 NPU 服务 PSS、CPU、FD、队列及模型释放；不出现持续无界增长。
5. **正式打包真机验证。** 前端版本、源码、协议和回归记录可追溯后，通过应用中心
   升级，保留现有配置、口令和 Workflow。再次使用原画布执行验收。

Detached 最终验收另要求：没有 Roboflow 登录和云 Key，本地 Workflow / RKNN 模型
仍能完成上述原画布 RTSP Run。浏览器仍从官方站点加载画布时，不能据此承诺浏览器
离线编辑；已部署本地流程的离线运行是另一个独立验收项目。

## 官方参考

- [Detached 与 Cloud connected 模式](https://docs.roboflow.com/workflows/tutorials-1/hello-world)
- [官方包装页源码](https://github.com/roboflow/inference/blob/main/inference/core/interfaces/http/builder/editor.html)
- [视频处理、RTSPSource 与持续会话](https://docs.roboflow.com/workflows/deploy/video-processing)
- [原画布测试与图像输出](https://docs.roboflow.com/workflows/build/test-a-workflow)
