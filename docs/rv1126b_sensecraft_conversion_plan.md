# RV1126B Workflow 与 SenseCraft 模型转换接入方案

评估日期：2026-09-16。范围：源码检查、官方文档核对和真机只读检查。
本轮未安装转换工具、未提交云端转换、未更改设备配置。

## 结论

采用 SenseCraft 在线转换，设备负责上传、任务管理、下载、校验、登记和 NPU 推理。
打开 Workflow 画布不强制 SenseCraft 登录；提交在线转换时才要求有效授权。
复用设备已有 OAuth 设备授权，不建立另一套 Workflow 专用 SenseCraft 登录。

## 本地转换可行性

真机为 AArch64、Buildroot 2023.02.6、Python 3.11.6。此次采样 MemTotal 为
2033652 KiB（约 1.94 GiB），MemAvailable 为 1203556 KiB（约 1.15 GiB），
/userdata 可用空间约 1.7 GiB。采样不能代表长期空闲资源。

Rockchip RKNN-Toolkit2 2.3.2 有 ARM64 CPython 3.11 安装包，所以不能断言
ARM64 设备绝对不能转换。但完整 Toolkit 的官方依赖包含 PyTorch、ONNX、
ONNX Runtime 等，重新引入这些依赖会偏离当前轻量推理目标。图优化、编译和量化
还会与视频处理争用 CPU、内存和存储；此轮未测转换峰值，不能承诺当前板卡可稳定转换。
设备现有 Lite2/Runtime 面向推理，不提供完整 ONNX 编译能力。
Seeed 针对 reCamera Pro 的指南也指定在外部 PC/WSL 转换。

建议不向设备安装完整 Toolkit；支持 SenseCraft 云端和未来可替换的局域网转换服务。

官方依据：

- [Rockchip ARM64 Python 3.11 依赖](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknn-toolkit2/packages/arm64/arm64_requirements_cp311.txt)
- [Rockchip 2.3.2 更新记录](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/doc/changelog-2.3.2.txt)
- [reCamera Pro 模型转换指南](https://wiki.seeedstudio.com/recamera_pro_rknn_model_conversion/)

## 已有能力与缺口

设备 GET /cgi-bin/entry.cgi/sensecraft/session 此次返回 authorized。
这说明设备已有登录授权，不代表本轮验证了云端转换配额或任意模型转换权限。

现有 Modelmanage/SensecraftPanel 和 SensecraftAPI 已实现：

- OAuth 设备授权、会话状态、刷新、注销；前端状态响应不携带 access/refresh token。
- ONNX 上传，可附带 dataset_file ZIP；现有转换参数 framework_type=9、device_type=40。
- POST /v1/api/create_task、GET /v1/api/train_status、转换记录与下载入口。
- 通过内置模型下载任务获取 RKNN。

需要补齐的部分：

1. 共享的授权与转换任务服务。现有转换轮询和进度主要由前端组件管理，不能直接当作
   可跨浏览器刷新、应用重启恢复的 Workflow 任务管理器。当前 Nginx 转发配置也不能
   直接等同于带设备会话鉴权的转换服务；需由系统后端读取已有授权并访问固定云端接口。
2. 统一模型资产登记。内置模型下载结果不会自动成为 Workflow 的模型包。
3. 动态模型授权。目前 appmgr/inference_auth.py 只接受安装包内声明的 bundled RKNN，
   并校验路径、大小和 SHA256；给共享目录增加一个文件不足以获得 NPU 加载权限。
4. Workflow 模型解析。目前 LocalModelStore 读取应用 models/ 下的 model.json 与 RKNN，
   只支持已有目标检测解码器匹配的模型，不能把任意转换产物直接标记为可用。

## 推荐交互

| 操作 | SenseCraft 登录要求 | 行为 |
| --- | --- | --- |
| 打开画布、编辑和保存流程 | 不要求 | 正常进入；可显示云端转换连接状态 |
| 使用已登记本地 RKNN | 不要求 | 按现有设备权限运行 |
| 导入已经转换好的 RKNN | 不要求 | 校验元数据和兼容性后登记 |
| 上传 ONNX 并在线转换 | 要求 | 检查并刷新授权；必要时打开已有登录组件 |
| 获取账号下的私有转换产物 | 要求 | 使用同一设备 SenseCraft 会话 |
| 转换完成后的持续推理 | 不要求 | 使用本地资产，不依赖云端会话 |

在应用卡片“更多”增加“模型管理”，复用内置 SenseCraft 的授权和转换组件。
该抽屉提供本地可用模型、ONNX 转换、转换任务及已有产物导入。
编辑入口可进行轻量状态检查，但匿名用户仍能直接进入画布。
登录成功后返回原转换任务，不清空已选择的文件和模型设置。

官方画布由 app.roboflow.com 托管。本地后端能提供模型目录、运行接口和错误信息，
不能假定可以直接在官方模型选择弹窗内植入 SenseCraft 登录或转换按钮。
首版将“准备模型”放在设备应用卡片内，准备完成后由原生画布选择本地模型。
设备端发布/启动预检列出缺失模型；已有工作流仍可保存，不因缺失模型而丢失编辑内容。

## 模型准备与运行流程

1. 提供 ONNX、模型类别及已知输入输出配置；INT8 可提供代表实际场景的校准数据。
   也可直接导入已有 RKNN 和配置，跳过转换。
2. 系统检查 SenseCraft 会话，刷新过期 access token；刷新失败才要求重新授权。
   账号或配额拒绝与登录过期分别报告。凭据保留在系统后端，不放进官方画布 URL、
   工作流 JSON 或 inference 应用配置。
3. 创建持久化本地任务，保存云端任务 ID、源文件摘要、目标平台、转换配置及状态。
   流式上传/下载，暂存目录使用有磁盘配额的 /userdata 路径；当前 /tmp 是 tmpfs，
   不适合缓存大 ONNX/校准包。先串行处理模型准备任务，限制缓冲区、超时和重试。
4. 后台跟踪云端任务。浏览器关闭后任务仍可恢复；超时先查询既有任务，避免重复提交。
   状态区分上传、排队、转换、下载、校验、就绪、需要登录及失败。
5. 下载后核对实际目标 RV1126B、运行时兼容性、文件摘要、输入输出张量、类别顺序、
   颜色/归一化、resize/letterbox 和后处理约定。字段缺失要求补充，不能从文件名猜测。
   首版限当前已适配的 YOLO 目标检测输出；不支持的分类、分割、关键点等单独报告。
6. 注册不可变模型资产和版本；建立“Workflow model_id → 精确模型版本”的绑定。
   appmgr 从经过校验的绑定生成进程级 NPU 授权，包含路径、SHA256、输入规格和资源限制。
   保留现有 bundled 模型兼容；运行进程不能任意放行文件路径或自行伪造授权。
7. Workflow 模型列表展示校验且授权就绪的模型。先完成加载/样例推理验证，再开放部署。
   对需更新进程授权的绑定，第一版通过现有应用管理器受控重启应用生效；不直接热替换
   正在运行的模型。仅改工作流模型选择不隐式启动转换任务。

绑定文件由系统管理，独立于应用安装目录，应用升级保留。模型 ID 冲突要求明确解决，
禁止用一个不同模型冒充相同 Roboflow model_id。原始 Workflow JSON 可保留，运行前解析
绑定到本地模型；模型作为动态输入时，预检使用本次实际参数进行解析。

模型转换就绪不等于自动启用内置推理。下载和验证也不应抢占内置推理的模型配置。
删除模型时检查内置和 Workflow 引用，使用中的资产先解除引用。

## Roboflow 模型来源

SenseCraft 登录只解决转换服务的访问权限。它不能把任意 Roboflow model_id 变为 ONNX，
也不替代 Roboflow 的权重下载授权。官方资料说明，权重下载还取决于模型类型和账号权限。
参见 [Roboflow 权重下载说明](https://docs.roboflow.com/deploy/download-roboflow-model-weights)。

首版优先支持用户提供 ONNX、复用 SenseCraft 已完成产物、导入现成 RKNN。
第二阶段再接入具备明确导出权限的模型来源；如获得的是 .pt，导出 ONNX 也放在外部服务。
不承诺官方画布里所有模型都可在 RV1126B 上转换和执行。

## 实施阶段与验收

第一阶段完成共享登录/后台转换任务、模型资产绑定授权、卡片模型管理、Workflow 目录和
部署预检，跑通“一份受支持 ONNX → SenseCraft → RKNN → 原生画布选择 → NPU 真实结果”。
现有 SenseCraft HTTP 接口可作为转换提供方，无需另起通用转换服务器。
先核验一份实际产物的工具链版本、输出布局与元数据；现有 UI 未提交独立 target_platform
或 Toolkit 版本字段，不能仅凭 device_type=40 就省略产物兼容性检查。

第二阶段增加可授权导出模型的自动获取、更多解码器、任务复用和模型版本切换。

验收应覆盖：匿名编辑、本地推理、登录后恢复任务、刷新/关闭页面/重启后任务恢复、
转换失败和凭据失效、产物损坏/不兼容、模型列表更新、重启后绑定保持、升级不丢模型，
以及上传/下载/持续推理阶段的内存采样。设备不新增 PyTorch、ONNX Runtime 或完整 Toolkit。
