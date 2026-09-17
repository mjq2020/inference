# RV1126B 推理应用 0.3.5：使用、依赖与构建

此目录构建独立 `inference-rv1126b` 发行包，不运行仓库根目录的
`setup.py`，不安装完整 `inference` / `inference-models` 发行包。
训练、模型导出、ONNX 转换和 RKNN-Toolkit2 均留在开发机或转换服务。

## 固件契约

- CPython 3.11，AArch64；二进制依赖选择 manylinux 2.17 / 2.27 / 2.28
  AArch64 wheel，需要兼容 glibc（本项目目标固件为 glibc 2.38）。
- 固件提供 NumPy **1.23.5**、OpenCV **4.6.0**；应用不得覆盖它们。
- 固件提供 `kit.runtime.remote.RemoteRknnSession`、
  `kit.adapters.official.OfficialFrameSource` 及相应相机 / NPU 服务。

可在已具备固件环境的设备上执行只读检查：

```sh
python3 deploy/rv1126b/platform_probe.py --require-device
```

应用中心 `manifest.python.imports` 使用 `inference_edge_probe`。该模块先设置
`INFERENCE_RUNTIME_PROFILE=rv1126b`，再导入 API、原 Workflow 引擎和支持的节点；
同时检查 NumPy/OpenCV 版本并拒绝重型模型框架。不要把 `inference.edge.api`
直接放在安装探针中，因为探针不传环境变量，而普通 `inference` 默认仍是完整模式。

## 原生 Workflow 画布入口

0.3.1 保留 Roboflow 原生画布及 Workflow JSON，设备端运行原 `ExecutionEngine`。
统一入口是设备 **App Center → Inference Workflows** 卡片。配置、启动、停止、
结果和编辑均从卡片进入；旧 `/device` 自动跳转到卡片结果窗口。
`/` 和 `/build` 保留直接连接入口，早期自建画布 `/legacy` 不再作为主要入口。

1. 在设备应用中心选择 **Inference Workflows → 更多 → 配置**，设置应用访问口令，
   将 `host` 设为 `0.0.0.0`。首次编排时将工作流设为“仅启动编辑服务”，保存并启动应用。
2. 选择 **更多 → 编辑工作流**，复制弹窗的完整连接地址，再打开原生画布。
   此入口沿用设备管理登录，自动换取临时连接授权，无需再次输入应用口令。
   官方画布在浏览器中联网加载，本地编排和发布无需 Roboflow 云端 API Key。
3. 点击官方画布顶部的运行位置（通常显示 **Locally**），在 **Running On**
   弹窗中选择 **Other**，粘贴刚才复制的完整运行地址，再点击 **Connect**。
   地址中的 `/ui/runtime/…` 路径也是连接凭据的一部分，不能只填 IP 和端口。
   浏览器询问本地网络访问权限时，允许官方页面连接此设备。
4. 按原有方法添加节点、连线、配置输入和输出。目标检测节点选择设备已安装的
   RKNN 模型；其他模型不会因节点在官方画布中可见而自动获得设备支持。
5. 点击 **Publish** 将流程保存到当前连接的设备。当前官方 `local` 页面的
   **Save** 草稿操作存在 `saveDraft` 前端错误，应使用 **Publish**。
   保存验收以设备 `POST /build/api/{id}` 返回 201，刷新或重新打开后仍可读取为准；
   不要仅凭画布上出现图形判断已保存。真机完整发布验收见下方记录。
6. 单图测试使用原生画布的测试面板。**当前官方本地画布的 RTSP / Webcam 预览存在
   无云端 API Key 时的前端阻断**，会显示 `Missing video preview configuration`，
   此时尚未向设备发起 WebRTC 初始化；设备访问口令不能补齐官方前端内部的云端配置。
   官方 SDK 的视频协议测试通过，不代表该画布入口已通过验收。
   原因与现有可用入口见[Workflow 视频预览说明](../../docs/rv1126b_workflows.md#原生画布的-rtsp--webcam-预览限制)。

运行地址绑定当前设备浏览器会话，登录满 **12 小时**、退出或应用重启后失效。
失效后重新登录、复制并连接；不要公开分享该地址。地址中使用临时授权，
不会放入长期设备口令。官方画布在浏览器中联网加载，Workflow 和 RKNN 推理在设备执行。

**Publish 保存不等于启动推理。** 回到应用卡片的配置窗口，从下拉框选择已保存的
Workflow，选择设备摄像头或 RTSP，填写参数和目标帧率，然后保存、启动。
RTSP 地址可包含用户名和密码，配置表单按密码字段显示。流程必须恰好声明一个图像输入。
卡片“查看结果”显示当前实例的图像输出与 JSON；若流程没有图像输出，只显示 JSON，
不会用设备摄像头画面替代 RTSP 结果。可视化需在 Workflow 中声明图像输出。
关闭结果窗口不会停止任务；点击卡片或结果窗口中的“停止”会关闭应用并释放模型和视频源。
保存新的 Workflow 定义不影响当前实例，重启应用后才使用新定义。

## 启动参数与持久化

参数由应用中心保存，均在应用重启后生效；运行中的应用保存配置后会重启，
相机任务随旧进程结束。停止状态下保存的参数在下次启动生效。

| 参数 | 默认值 | 作用与配置范围 |
|---|---|---|
| `host` | `127.0.0.1` | 设备本机；`0.0.0.0` 允许局域网访问 |
| `port` | `9001` | HTTP 端口，1024–65535 |
| `api_token` | 空 | 本应用页面/API 的访问口令，与设备管理密码独立 |
| `roboflow_api_key` | 空 | 可选，仅获取云端 Workflow 定义时需要 |
| `workflow_id` | 空 | 下拉选择本地 Workflow；空值仅启动编辑服务，非空时启动应用即运行流程 |
| `video_source` | `camera` | `camera` 为设备摄像头，`rtsp` 为网络视频流 |
| `rtsp_url` | 空 | RTSP/RTSPS 地址，密码类型配置，仅 RTSP 输入使用 |
| `workflow_fps` | `3` | 目标处理帧率，实际受 `max_fps` 与设备吞吐限制 |
| `workflow_parameters` | `{}` | JSON 对象字符串，例如 `{"confidence":0.4}`；名称须对应流程输入 |
| `max_fps` | `10` | 处理帧率上限，0.1–60；不是实际吞吐承诺 |
| `max_workflow_steps` | `128` | 展开嵌套后的节点上限，1–256 |
| `max_request_mib` | `32` | 请求体上限，1–64 MiB |
| `max_response_kib` | `8192` | 业务响应上限，64–16384 KiB |
| `max_image_pixels` | `8847360` | 输入累计像素预算，65536–8847360 |
| `max_video_output_pixels` | `921600` | WebRTC 预览像素上限，307200–2073600；默认 1280×720，仅缩放预览 |

非本机监听且 `api_token` 为空时拒绝启动。默认仅本机监听、空口令继续允许本机调用。
设备原生 Web 登录、SSH 密码、应用访问口令和 Roboflow API Key 各自独立。
API 可以通过 `Authorization: Bearer <设备口令>` 或 `X-Inference-Token` 认证；
原官方 SDK 默认请求体中的 `api_key` 也可使用设备口令。获取云端定义时，推荐在
应用配置中另设 `roboflow_api_key`，客户端仍使用设备口令认证。设备口令不会转发
给 Roboflow；配置云端 Key 也不会使未认证调用获得设备访问权。

原生 Web 使用设备登录会话保存配置：

```text
GET /api/app-center/v1/apps/inference-rv1126b/config
PUT /api/app-center/v1/apps/inference-rv1126b/config
Content-Type: application/json

{"values":{"host":"0.0.0.0","port":9001,"api_token":"用户设置的访问口令"}}
```

启动所选流程可在同一配置中设置非空 `workflow_id`、`video_source`，以及
`workflow_parameters` 字符串，例如 `{"confidence":0.4}`。应用中心验证类型与范围，
并按 `apply: restart` 持久化应用。不要添加内部代理认证头。
持久化目录来自 `kit.config.appdata_root()`，目标设备默认布局如下：

```text
/userdata/local/appdata/inference-rv1126b/
  config.json        # 应用中心保存的启动参数
  workflows/         # 原生 Builder 保存的定义和配置
  cloud-workflows/   # 按需建立的云端定义缓存
```

该目录位于版本化安装目录之外，升级时保留。**升级不会覆盖已保存的参数**：
例如旧版本保存的 32 节点或 1024 KiB 响应上限仍会生效，需要在应用中心主动调整。
Kit 合并后的有效配置覆盖 manifest 默认值及开发环境变量；模型目录固定为当前
安装目录的 `models/`，参数页面不能改写模型与数据路径。开发机运行
`inference_edge.py` 时仍可使用 `INFERENCE_EDGE_STORAGE_ROOT` 等环境变量。

应用中心模式下，非空 `workflow_id` 就代表启动应用时运行流程；不再需要单独的
`workflow_autostart` 开关。旧配置中该字段会被忽略。后台读取本地定义、解析嵌套流程，
冻结参数并打开选定视频源。成功处理首帧后才发送 Kit READY；输入、模型或执行错误
会使应用启动失败，并使用应用管理器已有的有界重试策略。HTTP 就绪不等于任务成功。
`GET /workflow-deployment` 可查看状态；原生 App Center 通过受设备登录保护的
`/api/app-center/v1/apps/{id}/workflows/runtime` 查看当前实例结果。

此集成需要同时部署 SDK `market/appmgr/workflow_ui.py`、对应 `server.py` 路由和
App Center React 前端。应用用 `x-workflow-ui: {"version": 1}` 声明契约。管理器仅从
该应用的 appdata 读取有界工作流列表，通过固定 loopback 端点读取结果、换取临时授权；
不在管理器进程内执行模型。应用停止后仍可列出已保存工作流并修改配置。

## 0.3.1 应用卡片验收

设备原生 App Center 已安装 0.3.1。卡片配置下拉选择、摄像头与 RTSP 运行、原生画布
临时连接、真实输出图像/JSON、停止后清除结果及资源释放均已真机验证。
原有五个 Workflow 文件保持不变，RTSP 模型调用使用固件 RKNN/NPU 服务。
测试用 RTSP 是设备子码流，不代表已验证用户尚未提供的外部视频流。

完整构建、截图、采样、错误恢复记录与测试结束状态见
[应用卡片真机报告](/home/dq/github/RV1126B_Linux_IPC_SDK/artifacts/inference-rv1126b-app-card-20260915/REPORT_CN.md)。

## 0.2.0 支持范围与验收状态

当前设备目录包含 **121 个节点版本条目**，包含同类节点的 v1/v2/v3；不是 121 种
模型。恢复了跟踪、越线计数、嵌套 Workflow、图像处理、可视化、查询表达式等原节点。
实际目录以 `/workflows/blocks/describe` 为准，细节及 API 示例见
[Workflow 支持说明](../../docs/rv1126b_workflows.md)。

模型推理仍限已安装、元数据与当前解码器匹配的 RKNN YOLO 目标检测模型。
应用不安装 PyTorch、TorchVision、ONNX Runtime、训练或本地模型转换工具，
也不以 CPU 模型执行作为回退。ONNX→RKNN 远程转换服务尚待用户提供 HTTP 契约；
当前转换 API 会明确报告未配置。

已完成的开发机验证包括：原 Engine 裁剪/检测适配/跟踪与计数、原官方 HTTP SDK
调用、原配置格式与旧版本数据读取、真实 WebRTC 协议和视频数据传输、自动启动与
停止清理、隔离 wheel 安装和重框架导入检查。软件模型管理器和模拟相机测试不计作
真实 NPU 或真实摄像头验收。

2026-09-15，RV1126B 真机通过设备原生 Web/JWT 应用中心安装并启动
`0.2.0-b469c6bc93c19ab9`，安装包包含 64 个 wheel。对应源码 wheel SHA256 为
`7061bb09c5713366fe8e8df930893ac03360fee33baae79304c023a4c44336e0`。
164 项应用测试、SDK 原生安装包校验及设备安装后的源码/依赖核对通过。

| 真机验收 | 实际结果 |
|---|---|
| 原生画布 | 官方 `local` 页面无需账户；Publish 返回 201，刷新重开保留；Run 返回 200、计数为 2 |
| 保存的原生示例 | `custom-workflow-2`，名称“RV1126B 原生检测与可视化”，检测→计数→框→标签共 4 个原节点 |
| 官方 HTTP SDK | 24 项通过；检测 v1/v2/v3 × 三种认证 × inline/local 的 18 组合一致，另覆盖参数/过滤、两张图批次、嵌套和错误 |
| 设备摄像头 | `/device` 使用 Kit 1280×720、帧率上限 3；累计处理 437 帧，暂停时帧数 381 保持 2 秒，恢复后增长至 437，停止释放模型 |
| 配置自动启动 | 应用中心保存配置后重启，PID 1938→2821、generation 17；目标流程状态 `running`，收到 23 帧；同期 NPU 记录 85 次完成调用、失败 0 |
| 官方 SDK WebRTC | ManualSource 收到 4 份检测 JSON/3 帧预览；MP4 收到 4 份 JSON/4 帧图像；旧协议收到 4 份 JSON/3 帧预览，结束均释放会话与模型 |
| 分辨率与坐标 | 1080p 视频输入可缩小为 720p WebRTC 预览，检测 JSON 尺寸和坐标仍对应原图 |

HTTP SDK 使用真实 1920×1080 测试图、`yolo8n/1` RKNN，输出两个犬类检测；
普通单图约 660–792 ms、首例约 1048 ms，双图批次约 1279 ms。这些时间包含网络、
引擎执行及图像序列化，不是纯 NPU 时延，也不能当作摄像头持续 FPS。

本次 PSS 观测如下，**都是指定阶段的短时采样，不是长期峰值或内存上界**：

| 场景 | 应用进程 PSS | 共享 `inferenced` PSS |
|---|---:|---:|
| 启动初始 | 82.18 MiB | 未在此项汇总 |
| 打开原生节点目录后 | 约 270 MiB | 未在此项汇总 |
| 直接重启并仅运行设备摄像头 | 216.13 MiB | 37.24 MiB |
| 同一活动 WebRTC 会话的最大同步样本 | 468.095 MiB | 37.205 MiB |

最后一项合计约 **505.30 MiB**。`inferenced` 是平台共享推理服务，不是应用的第二个
进程；不能把不同阶段的最大值相加，也不能将服务登记的模型内存预算再次加到 PSS。

验收结束保留用户原有口令、端口、三个已有流程及新增原生示例；恢复
`workflow_autostart=false`、`workflow_id=""`，应用 HTTP 服务保持运行，摄像头已停止。
原始证据位于
[本次验收目录](/home/dq/github/RV1126B_Linux_IPC_SDK/artifacts/inference-rv1126b-native-workflows-20260915)，
包括 `test-workflow-sdk.json`、`native-run-result.json`、`camera-*.json`、
`autostart-result.json`、`webrtc-acceptance.json` 和 `webrtc-running-memory.json`。
这些文件保存在本次工作区，复用构建时应另行归档。

## 文件职责

| 文件 | 用途 |
|---|---|
| `requirements.txt` | CPython 3.11 完整锁定运行依赖；不含系统 NumPy/OpenCV |
| `source_files.txt` | 审查过的源码清单，保留原 Workflow 引擎和当前可用节点 |
| `build_wheel.py` | 只读取源码，生成确定性的应用 wheel 和平台派生依赖 wheel |
| `download_wheels.py` | 下载 ARM64 依赖并构建上述两个 wheel |
| `audit_wheels.py` | SDK tag 规范化、ARM ELF / 禁包 / 依赖闭包检查和 SHA256 清单 |
| `smoke_wheel.py` | 从临时目录安装源码 wheel，验证 API、实际裁剪工作流及区域判断 |
| `platform_probe.py` | 只读检查固件 ABI、图像库和 kit 导入契约 |
| `build_app.py` | 使用当前 SDK 的原生打包器生成应用中心安装包 |
| `../../build_scripts/download_fonts.py` | 预取原可视化字体和许可证，验证 SHA256 |

源码清单排除了 `inference/models`、`inference/core/models`、完整 HTTP/SDK
客户端、企业实现、训练代码、测试和未支持的 Workflow 模型节点。
构建器还检查当前节点目录的每个模块是否已列入源码清单。新增设备功能时应显式扩展清单，并重新执行隔离安装验证。源码目录中存在的
`core/utils/onnx.py` 仅保留通用 NumPy / session 辅助函数，不安装 ONNX Runtime。

## Supervision 与固件 OpenCV

上游 `supervision==0.29.1` 的元数据要求 PyPI 分发包 `opencv-python`。
固件的 `cv2` 并不等于一个需要重新安装的 PyPI wheel。

构建器将原 Supervision wheel 派生为单独的 **`supervision-rv1126b==0.29.1`**
分发包，保持 `supervision` Python 源码、功能和许可证不变，只移除
`opencv-python>=4.5.5.64` 的 `Requires-Dist`，改为说明固件提供的 OpenCV。
派生包在 `RV1126B_ADAPTATION.json` 中记录原包名称、SHA256、删除的依赖及原因。
不创建虚假的 `opencv-python.dist-info`，不提供或覆盖 `cv2`。

NumPy 仍由固件提供；所有依赖对 NumPy 的版本约束由审计器统一检查为可接受
1.23.5。Shapely 固定为 **2.0.7**，用于按需执行区域判断；
`dataclasses-json` 固定为 **0.6.7**，用于原 SDK 的公共类型。

## Buildroot 精简标准库与 NetworkX

目标固件不提供 `_bz2`。上游 `networkx==3.4.2` 在包初始化时导入 `bz2`，
虽然 Workflow 只使用图算法，也因此无法启动。派生分发包
**`networkx-rv1126b==3.4.2`** 仅将 `networkx/utils/decorators.py` 中 `.bz2`
图文件的打开函数改为调用时导入 `bz2`。图算法和普通图 IO 保持不变；实际请求
压缩图 IO 而固件缺少 `_bz2` 时，明确抛出错误，不将压缩文件作为普通文件读写。
源码差异及原 wheel / 修改前后文件 SHA256 记录在 `RV1126B_ADAPTATION.json`。
不伪造标准库模块，不修改系统 Python。

派生包与上游同名 Python package 不应安装到同一个应用依赖目录；升级时使用
应用中心新建的版本化依赖目录，避免原 `networkx.dist-info` / `supervision.dist-info`
与派生元数据同时存在。NetworkX 此版本没有顶层 `lzma` 导入；隔离验证还会模拟
缺少 `bz2`、`_bz2`、`lzma`、`_lzma`，确认保留功能可在裁剪标准库上运行。

## 字体、WebRTC 与应用依赖隔离

原可视化字体注册表用于 Label、Rich Label、Text Display 等节点。构建前执行：

```sh
python3 build_scripts/download_fonts.py
```

下载器从注册表的固定地址取得字体和 `OFL.txt` 许可证并核验 SHA256。
构建 wheel 时再次检查完整字体集合、许可证、来源及摘要；缺失或不匹配即失败，
不在设备运行时下载或换用不确定的系统字体。字体及来源记录随源码 wheel 打包。
自定义预取目录时，给 `download_fonts.py` 传 `--target-dir`，并给后续
`download_wheels.py` / `build_wheel.py` 传相同的 `--fonts-dir`。

WebRTC 使用锁定的 `aiortc`、`av` 及其 CFFI/加密依赖，媒体模块按需加载；
PyAV wheel 自带媒体库也属于 ABI/ELF 审计范围。普通 HTTP 健康检查不能证明
WebRTC 二进制链已在设备加载成功。

Kit 启动器可能先导入固件自带的 Pillow/CFFI。应用入口 `app.py` 在导入业务库前
检查本版本私有依赖的优先级；必要时以前置私有 `PYTHONPATH` 在同 PID 重新执行
启动器，清除先前缓存的旧模块。它保留原应用启动身份和资源授权，不修改固件库，
也不自行创建 NPU 授权。更换 SDK 或启动方式后应重新验证这一初始化路径。

## 重建 ARM64 wheelhouse

使用开发机隔离的 Python 3.11 环境，至少安装 `pip==25.3` 和 `packaging==24.2`。
不要将下面的开发依赖安装到设备系统 Python。

```sh
python deploy/rv1126b/download_wheels.py --output /tmp/inference-rv1126b-wheelhouse
python deploy/rv1126b/audit_wheels.py /tmp/inference-rv1126b-wheelhouse --normalize
```

下载器显式使用 `--no-deps` 和锁定清单，所以不会隐式下载 NumPy、OpenCV、Torch
或模型转换框架。它不继承宿主机的额外 CUDA package index；可通过
`--index-url` 指定包索引。下载后的原始 wheel 和源码内容决定输出；
同一输入重复构建时 wheel 的时间戳、文件顺序、RECORD 和 SHA256 保持一致。

SDK 的 wheel 校验要求文件名的 tag 与 WHEEL 的单个 Tag 完全一致。
审计器将纯 Python `py2.py3-none-any` 规范化为 `py3-none-any`，并将已有的
compound platform tag 写为对应的单个 WHEEL Tag。它重写 RECORD、保存原文件名
和原 SHA256，不改变任何二进制 ABI；不兼容的 CPython / 架构标签直接报错。
SDK 不支持 wheel 的 `.data` 安装布局；锁定的 FontTools wheel 唯一此类文件是
非运行时的 `ttx.1` 手册页，规范化时仅删除该手册页并记录来源，其他 `.data`
文件一律报错，不静默丢弃运行时资源。

审计通过后得到 `SHA256SUMS` 与 `wheelhouse-audit.json`。后者包含每个包的版本、
Requires-Dist、压缩/展开大小及目标固件契约。必须连同 wheelhouse 一起保留，
后续离线重装使用同一组文件及摘要。应用中心通过 runtime-v2 将这些 wheel
安装到应用自己的依赖目录；不要对系统 Python 执行 pip install。

仅修改应用源码时，无需重新下载依赖：

```sh
python deploy/rv1126b/build_wheel.py --output /tmp/inference-rv1126b-wheelhouse
python deploy/rv1126b/audit_wheels.py /tmp/inference-rv1126b-wheelhouse
```

## 开发机上的隔离安装验证

先准备与固件 NumPy/OpenCV 版本一致的开发机测试环境，以及清单中的对应
宿主机架构依赖。不要把 AArch64 二进制 wheel 安装到 x86_64 测试环境。

```sh
python deploy/rv1126b/smoke_wheel.py /tmp/inference-rv1126b-wheelhouse
python deploy/rv1126b/smoke_wheel.py /tmp/inference-rv1126b-wheelhouse --unwritable-home
```

验证器只把应用源码和派生 Supervision / NetworkX wheel 装入临时目录，以宿主机测试环境
提供其他依赖。子进程在仓库之外执行，断言 `inference`、`inference_sdk` 和
`supervision` 均来自临时安装目录，检查 API / OpenAPI、真实裁剪 Workflow 和
Shapely 区域判断，并确认没有加载重型框架。
`--unwritable-home` 额外模拟应用中心安装探针的 `HOME=/nonexistent` 环境；
Matplotlib 可以回退到临时缓存目录，检查不依赖可写的用户主目录。

这是源码分发及依赖边界验证。上文记录了本次固件的真机验收；更换固件、SDK、
模型或二进制依赖后，仍需重新检查实际动态库加载、摄像头生命周期、RKNN 结果和 PSS。

## 实时预览叠加与 Workflow 输出图像

从 0.3.8 开始，两个入口各自承担不同用途：

- 设备“实时预览”：浏览器 Canvas 绘制 Workflow 输出的结构化检测框与标签。
  摄像头工作流叠加到设备主码流，并可参与“全部可绘制应用”。RTSP 工作流使用
  自己的原始画面与同一帧检测数据，在前端叠加；不与设备主摄像头结果混合。
  “AI 叠加”开关只控制浏览器绘制，不改变工作流。
- 应用卡片“查看结果”：显示 Workflow 可视化节点产生的图像流，保留节点的
  绘制风格；多个图像输出可选，支持暂停观看，关闭窗口不停止应用。
- 应用卡片“输出数据”：仍可查看、复制 Workflow 的结构化 JSON。

实时预览需要 Workflow Outputs 包含结构化检测输出，例如 `model_output`；
仅返回可视化图片时不能从图片可靠还原检测坐标，不加入前端叠加来源列表。
目前浏览器适配检测框、类别和置信度；Workflow 其他视觉效果仍通过卡片的
原始图像输出完整呈现。坐标必须为根输入坐标，尺寸与原始输入匹配。
`coordinates_system=own` 的检测仅在其 `parent_id` 关联当前原图输入且尺寸一致时
参与叠加，裁剪坐标不直接用于整图。空检测帧会清除旧框。
应用配置“显示设置 → 在实时预览中显示”保留当前浏览器的可见性偏好。

接口复用设备登录认证和正在运行的推理，不需要额外 Roboflow API key，
不创建第二份模型、视频源或推理管线：

- `/api/app-center/v1/apps/{app_id}/workflows/runtime` 默认返回轻量元数据，包括
  图像输出 `preview.outputs` 和可叠加检测输出 `preview.overlay_outputs`。
- `include_overlay=true` 按需返回归一化检测框（至多 16 个输出、共 128 个框），
  不包含 Workflow 可视化图片。摄像头只传结构化结果；RTSP 同时返回这帧的原图 JPEG，
  保证画面与坐标对应。数据过期、管线切换或该帧无输出时清除旧结果。
- `/api/app-center/v1/apps/{app_id}/workflows/preview?pipeline_id=...&output=...`
  为应用卡片提供 JPEG/PNG 图像，支持 ETag/304、425 等待、409 管线失效。
  代理只读取固定 loopback 服务，应用凭据不传给浏览器。

设备只缓存最新结果的图像引用，按需解码至多两张图像，每张上限 4 MiB。
RTSP 原图仅在观看时、在 Workflow 可能修改输入之前编码，观看租约 3 秒，
总像素不超过 1280×720。查看器串行限速取图；离页取消请求并释放显示资源。
工作流本身声明的可视化节点始终按原定义执行。

## 生成应用中心安装包

应用卡片使用随包携带的 `icon.jpg`（Roboflow 官方 Workflows 发布图），
由 manifest 的 `icon` 字段声明，打包时纳入文件校验清单。设备通过本地图片接口
提供展示，不请求外部图片地址。素材出处见 [ICON_SOURCE.md](ICON_SOURCE.md)。

准备已通过审计的 wheelhouse 和包含模型元数据的 RKNN 模型目录，使用目标 SDK
提供的打包规则：

```sh
python3 deploy/rv1126b/build_app.py \
  --sdk-root /path/to/RV1126B_Linux_IPC_SDK/project/app/recamera-pro-ext-api \
  --wheelhouse /tmp/inference-rv1126b-wheelhouse \
  --model-root /path/to/approved-rknn-models \
  --out /tmp/inference-rv1126b-package
```

保留安装包、对应 JSON 构建记录、源码 wheel 摘要和 wheelhouse 审计清单。
通过设备应用中心安装、启动和停止，使 `camera.frames` / `npu.rknn` 资源由平台授予
与回收。直接运行 Python 的导入测试不能替代带真实平台授权的应用验收。

### 大安装包的固件代理要求

本次安装包为 123179020 字节（约 117.47 MiB，通常显示约 118 MiB），没有超过应用
中心 200 MiB 包体限制。但旧固件的 `/_jwt_verify` 内部鉴权子请求继承 100 MiB
请求体限制，会在 JWT 验证前拒绝这个上传，即使应用中心上传 location 已允许 256 MiB。

固件权威配置
[`common_relay.conf`](/home/dq/github/RV1126B_Linux_IPC_SDK/project/app/recamera_web/recamera_web_backend/ipcweb-env-rv1126b/etc/nginx/common_relay.conf:94)
已在 `location = /_jwt_verify` 中设置 `client_max_body_size 256m`，与上传入口对齐。
`internal`、`auth_request`、JWT 验证及不转发请求体的 FastCGI 规则保持启用。
这是固件 nginx 配置的兼容要求，不是 Inference API 的 `max_request_mib` 参数。

本次使用真实 nginx 的两项回归验证了该行为，并通过正常 JWT 应用中心安装。
回归源码为
[`test_nginx_auth_upload_limit.py`](/home/dq/github/RV1126B_Linux_IPC_SDK/project/app/recamera_web/recamera_web_backend/tests/test_nginx_auth_upload_limit.py)。
在其他固件部署此包前，确认其权威配置已包含相同修复；重刷固件时也应保留，
不通过关闭 JWT、伪造内部认证头或绕过应用中心安装来处理上传失败。
