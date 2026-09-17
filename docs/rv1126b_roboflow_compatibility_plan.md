# RV1126B 与原生 Roboflow Workflows 兼容方案

调研日期：2026-09-15。源码基线为 `3e69d06b2` 及当前工作区的 RV1126B 改动；Inference 版本标识为 `1.5.2`，Execution Engine 为 `1.15.1`，最近一次安装的设备应用为 `0.1.3`。本轮查阅官方在线文档并审阅源码，没有修改运行代码、安装应用或重启设备。本文是待实施方案，不能作为原生画布已经通过真机测试的证明。

建议将目标明确为：使用原生 Roboflow 画布和原 Workflow 定义，在 RV1126B 上通过原 Execution Engine 编排、通过 RKNN 执行已适配模型。设备继续不安装 PyTorch、训练模块、ONNX Runtime 或模型转换工具。已有的“RV1126B 工作空间”是之前新增的编辑器，应退出主要编排入口；已有流程先备份、转换保存格式并验证可重新打开，再迁移入口。

**官方流程及其原理**

官方提供两条入口：Cloud connected 在 Roboflow 工作空间创建流程，通过 “Running on” 选择本地 Inference Server；Detached 从开发模式服务器的 `/build` 进入原生画布，无需 Roboflow 账户或 API Key。Detached 不提供云模型库、云部署和监控等账户功能，访问控制由部署方提供。[官方入门教程](https://docs.roboflow.com/workflows/tutorials-1/hello-world)

因此，需要修正此前报告中“`/build` 必须有 Roboflow 账户”的说法。此前直接打开不带连接参数的远程页面看到登录提示，不能证明正式 Detached 模式必须登录。另一方面，本仓库的 `/build` 是一个包装页，通过 iframe 加载 `BUILDER_ORIGIN/workflows/local`，传递 `serverUrl`、`csrf`；这也不能证明画布资源完全离线可用。设备断网执行与浏览器断网编辑是两个验收项目。[上游包装页](https://github.com/roboflow/inference/blob/main/inference/core/interfaces/http/builder/editor.html)，本地对应 `inference/core/interfaces/http/builder/editor.html:40`。

画布负责添加节点、配置参数及建立连接，实际程序是 JSON 中的 `version`、`inputs`、`steps`、`outputs`。例如 `$inputs.image` 引用输入，`$steps.detector.predictions` 引用检测结果。节点类型带有版本；界面布局与运行定义需要一起保存，但布局不决定执行顺序。[定义语法](https://docs.roboflow.com/workflows/developer-guide/developer-guide/definitions)

执行引擎检查节点类型、输入及连接，按依赖关系运行有向无环图，管理分支、裁剪产生的批次对应关系和最终输出。因此“同一 Workflow 能否在设备运行”取决于引擎、节点和模型能力，而不是重新绘制画布。原引擎具有并发执行能力；设备可以降低并发，保留数据依赖和输出语义，代价主要是吞吐量变化。[执行引擎说明](https://docs.roboflow.com/workflows/developer-guide/developer-guide/execution-engine)

用户操作应保留 Create → Add a Block → 参数/连线 → Test Workflow → Save → Deploy。云端 Save 保存草稿，Publish 才更新发布版本；设备的持续运行也应绑定明确的已部署版本，编辑或保存草稿不应悄悄切换摄像头任务。[编排与保存说明](https://docs.roboflow.com/workflows/build/build-a-workflow)

图片可通过标准 `InferenceHTTPClient.run_workflow()` 执行，也可提交完整定义，或用 workspace/workflow ID 运行已保存流程。当前官方视频指南使用 `client.webrtc.stream()`；一个持续会话支持跟踪、计数等跨帧状态。浏览器的 Test Workflow 视频预览还要求流程声明图像输出。[部署说明](https://docs.roboflow.com/workflows/deploy/deploy-a-workflow)、[视频处理](https://docs.roboflow.com/workflows/deploy/video-processing)、[浏览器测试说明](https://docs.roboflow.com/workflows/build/test-a-workflow)

**当前实现与目标的差距**

以下发现来自当前工作区；之前自建画布的真机成功不能替代这些协议的兼容验证。

| 能力 | 当前状态与需要完成的工作 | 源码定位 |
| --- | --- | --- |
| 原生画布 | `/build` 已保留包装入口，但首页是新增编辑器；未完成真实原生画布创建、保存、图片运行的闭环 | `inference/edge/browser.py:270`、`inference/edge/static/builder.js:14` |
| 标准 SDK 单图 | SDK 默认发送 `use_cache`、`enable_profiling`；edge 请求模型禁止额外字段且未定义二者，会返回 422。应恢复请求、响应及错误语义，而非要求修改 SDK | `inference_sdk/http/client.py:2549`、`inference/edge/api.py:101` |
| 本地保存格式 | 原本地加载器解析 `json.loads(saved["config"])["specification"]`；新增编辑器保存顶层 `specification`。存储层可以保留两种数据，但编辑和加载逻辑尚未打通 | `inference/core/roboflow_api.py:1692`、`inference/edge/static/app.js:257`、`:272` |
| 按 ID 执行 | 原服务支持 `/{workspace}/workflows/{id}`，`workspace=local` 读取本地流程；edge 只有提交定义的 `/workflows/run`。云端解析、版本选择与缓存也未接入 | `inference/core/interfaces/http/http_api.py:2317`、`inference/core/roboflow_api.py:1632` |
| 画布发现和调试 | 已有节点描述、schema、校验、动态输出；缺 `describe_interface`，错误结构简化，可能影响节点定位和错误显示 | `inference/edge/builder.py:70`、`inference/core/interfaces/http/http_api.py:2293` |
| 摄像头与视频 | 当前为自定义单摄像头协议；原 streaming SDK 的 URL、请求、响应不同；新版 `/initialise_webrtc_worker` 缺失 | `inference/edge/api.py:109`、`inference_sdk/http/client.py:2956`、`inference_sdk/webrtc/session.py:1271` |
| 视频预览 | 当前明确禁止摄像头图像输出，不符合原画布视频测试要求；需要有界图像/视频输出。Box/Label 用于常见检测预览，协议本身也允许直接输出原图 | `inference/edge/limits.py:68` |
| 视频状态 | 保留每流复用 Engine，但独立 `WorkflowVideoMetadata` 输入未按原流处理器注入；采集时间、处理时间及丢帧后的帧序语义也需核对 | `inference/core/interfaces/stream/model_handlers/workflows.py:140`、`inference/edge/video.py:204` |
| 节点范围 | 10 类实现是人为审核白名单，不是引擎上限；标准检测仅 v1，缺 v2/v3、分类、基础可视化、跟踪等 | `inference/core/workflows/core_steps/loader_rv1126b.py:163` |
| 检测语义 | 原 v1 默认 `max_detections=300`，设备版改为 32，原 JSON 未显式设置时结果会变化；应恢复默认或明确报资源限制，不静默改结果 | `inference/core/workflows/core_steps/models/rknn/v1.py:93` |
| 访问认证 | 现有原画布授权路径和 CORS 请求头范围不足以覆盖新增标准路由；必须一起适配 | `inference/edge/browser.py:76` |

不能简单把模型节点 v2/v3 的类型名替换成 v1。它们的输出及参数有所变化，例如新版输出的 `model_id`、v3 的置信度模式，均需要分别实现或明确拒绝。模型包未提供最佳阈值时，不应把 `best` 偷换成固定阈值。

**拟采用的分层实现**

| 位置 | 负责内容 |
| --- | --- |
| 用户电脑浏览器 | 原生 Roboflow Builder、编排、调试和 Deploy；画布的浏览器内存不会成为设备 Python 进程内存 |
| RV1126B 应用 | 兼容 HTTP API、本地定义/版本存储、原 Execution Engine、轻量节点、RKNN 模型管理、摄像头会话、必要的预览输出 |
| 设备 Kit 与系统服务 | 正式摄像头取帧、RGA 转换、NPU 调度、应用生命周期和授权 |
| 另一台转换机器 | ONNX → 目标 RV1126B 的 RKNN、量化和模型元数据；由用户后续提供 HTTP 接口 |
| 可选 Roboflow 云连接 | 使用真实工作空间、获取有权限的 Workflow/版本；不作为本地已部署流程持续运行的必要条件 |

首先以 Detached 原生画布完成本地闭环，再兼容云端 “Running on” 接入同一执行服务。正式验证应抓取原画布实际请求，不能仅根据猜测扩展 API。保留原节点描述、接口 schema、错误上下文、输入参数和输出结构；`use_cache`、预览、profiling、`disable_sinks` 等要区分具体功能，不能只接受字段后忽略其语义。支持的节点与版本通过能力接口和部署前检查明确报告。

浏览器认证要兼容原生 `X-CSRF` 交互及标准 SDK 的认证方式，同时将设备访问口令与 Roboflow 云 API Key 分开处理，不能将设备口令用于访问云端。当前 SDK 已支持 Authorization header；本地 Detached 不需要云密钥。启用云流程获取或远程转换时，应用的网络出口配置也需允许实际服务端点。

`http://192.168.66.80:9001/build` 不等同于官方示例的 `localhost`。Chrome 的本地网络访问机制需要用户授权且依赖安全上下文，单独增加 CORS/旧 PNA 响应头不足以解决。需实测当前浏览器的 iframe 安全上下文、授权、模型列表和执行请求；根据结果采用可信 HTTPS 入口或验证通过的官方 HTTPS 顶层接入方式，不把关闭浏览器安全功能当作正常使用步骤。[Chrome 官方说明](https://developer.chrome.com/blog/local-network-access)

模型层应在原标准 Model Block 后端解析 `model_id`，映射到已安装 RKNN 包；用户继续使用原节点、参数和下游连接。转换产物至少包含目标芯片、RKNN 兼容版本、文件摘要、任务类型、输入布局/尺寸/颜色/归一化、输出张量、类别顺序和后处理信息。仅返回一个 `.rknn` 文件不足以保证可运行；当前后处理只支持约定的 YOLO 输出，也不能据文件扩展名宣称支持任意模型。

转换应发生在模型准备/部署阶段。流程运行时加载已校验的产物，保持本地 NPU 执行。转换结果安装需要符合当前 Kit/NPU 服务的模型路径和摘要授权规则；不能将下载到任意目录等同于正式注册成功。远程接口未提供前，仍可使用已安装模型完成原画布与 API 兼容工作。

**节点恢复范围**

| 类别 | 建议策略 |
| --- | --- |
| 已有检测、裁剪、过滤、表达式、属性计数、ContinueIf | 保留原实现和选择器；补默认值、错误及完整调用协议回归 |
| Box/Label 可视化、字符串模板、JSON 解析、SwitchCase 等 | 优先加入；解决官方入门流程也需要的可视化能力 |
| 灰度、阈值、模糊、形态学、轮廓、尺寸/距离测量等 | 按 NumPy/OpenCV 依赖逐类审核，限制中间图像内存；不必恢复模型框架 |
| 跟踪、越线计数、区域停留等 | 保留跨帧状态，验证真实时间和帧序；为失踪目标记录、轨迹、缓存增加明确容量与回收机制 |
| 分类、分割、关键点和多模型串联 | 分任务实现 RKNN 元数据、后处理和原输出 kind；用实际转换模型验证，不能只加类型别名 |
| Webhook、文件输出等 sinks | 后续按应用需求接入实际权限、超时和有界队列；保留画布测试默认禁用及 Enable Sinks/disable_sinks 控制，不能将所有预览强制禁用 |
| 训练、设备端转换、依赖重型模型框架的节点、任意 Python | 不进入基础设备发行版；涉及远程能力时单独声明，不自动改为云推理 |

本轮在不安装 Torch/ORT 的宿主轻量环境对 32 个候选进行了导入检查，30 个可导入。它只表明部分节点可在轻依赖环境装载，不证明行为正确、已开放或已经通过真机测试。

**低内存策略与功能取舍**

1. 保留 NumPy、OpenCV 和必要序列化/图执行依赖，删除的重点仍是训练与无关模型后端。CPU 仍要完成裁剪、过滤、后处理和编排；NPU 承担神经网络计算。
2. 设备运行一个服务进程、一个默认视频任务，节点默认串行，按需导入实现。元数据 schema 也会触发导入，应研究静态描述缓存或把描述与执行依赖分离，避免打开节点列表就加载所有实现。
3. 视频执行实例按会话复用，HTTP 独立请求隔离状态。队列、裁剪总像素、模型调用次数、输出图像和缓存都有容量限制；超限返回具体节点及原因，不静默截断有效结果。
4. 保留预览能力，但限制分辨率、编码帧率和缓冲数量；只在有订阅者时编码预览，慢客户端淘汰旧预览帧。流程明确要求的图像处理仍须执行，不能因无人观看而跳过有下游依赖的节点。
5. WebRTC 媒体层是独立适配工作。优先复用平台已有硬件媒体能力，测量所需库与会话的增量内存，限制媒体 worker 数量。若需开发机媒体桥接才能满足预算，必须说明额外组件和网络路径；它不等于设备已原生支持 WebRTC。
6. 当前单模型会话在 A→B 模型之间会卸载重载，多模型流程可能可执行但性能很差。后续按实测内存允许有限的多个小模型常驻并串行使用 NPU；不预先承诺任意数量模型。部署检查需给出模型总需求与可接受的运行策略。
7. 取帧尽早判断暂停和丢帧，减少不必要转换；及时归还 DMA，停止流程清理模型、媒体任务和状态。保留 Kit 的 READY 与退出流程。

2026-09-14 的既有测试是参考基线：应用刚启动约 65–67 MiB PSS，访问/校验后约 174 MiB，短时摄像头流程约 188–193 MiB；共享 NPU 服务约 37 MiB 另计，停止后 Python 堆并未立即回落。恢复原画布不会自动消除引擎和节点元数据的内存成本。下一轮需分别记录应用 PSS、系统 NPU 服务、媒体服务及内核/驱动缓冲，不能仅以 Python RSS 代表总设备消耗。

该数据来自既有 [0.1.3 真机报告](/home/dq/github/RV1126B_Linux_IPC_SDK/artifacts/inference-rv1126b-workflow-ui-20260914/REPORT_CN.md)，不是本轮新增测量。下一版应以相同流程、分辨率和帧率与其对照；新增可视化、WebRTC、多模型的增量单独说明，不承诺未经测量的内存数字。

**实施顺序与验收**

| 阶段 | 工作及完成标准 |
| --- | --- |
| 1：原生单图闭环 | 原画布实际打开，显示真实 RKNN 模型；完成“输入 → 检测 → 过滤/计数 → Box/Label → 输出”。保存、刷新重开、标准 SDK 提交定义和按本地 ID 运行均成功；变量、节点错误可见；备份并迁移既有两份示例 |
| 2：原视频流程 | 按真实原画布及 `client.webrtc.stream()` 协议完成预览；分别验收浏览器/SDK 提供的视频输入与 Kit 本机摄像头。支持 `model_id` 简写时同步适配 SDK 自动生成的检测 v2，不能等到阶段 4；适配需保留的旧 stream SDK，支持启停、断连清理及视频元数据。跟踪/越线等使用固定视频核对状态，长时运行与重复启停验证内存、FD、NPU 会话和队列有界 |
| 3：部署与云端接入 | 兼容真实 workspace/workflow ID、发布版本与本地快照；增加所选流程、运行参数、自动启动策略。设备重启可恢复已部署版本，云连接中断时本地模型和快照继续运行；草稿保存与切换运行版本明确区分 |
| 4：模型与节点扩展 | 用户提供转换协议后接入转换及正式模型安装；在前面实际画布/SDK 必需的检测版本基础上，继续验证其他版本、分类等任务和多模型流程；发布节点类型、版本、参数与模型架构支持矩阵 |

跨阶段验收还包括：未改动官方 SDK 的调用、实际原画布的浏览器网络请求、裁剪后根坐标及批次对应关系、显式阈值与默认参数、RKNN 精度偏差、超限错误、预览与 sinks 行为，以及无 Torch/训练依赖的包检查。应固定服务器与 SDK 验证版本；云端画布可能独立更新，需要保留协议回归。

上述前三阶段中的原画布和视频流程都完成后，才应称为“保留主要 Roboflow Workflows 使用流程的 RV1126B 版本”。目前准确的状态是“原引擎的受限节点子集已在设备运行”；原生画布及完整 SDK/视频兼容仍待实施。
