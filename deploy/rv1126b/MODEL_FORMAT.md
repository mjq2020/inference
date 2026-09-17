# RV1126B 应用模型包

此应用只在设备上运行已经转换好的 `.rknn` 模型。模型加载、NPU 调度和模型显存由平台 `inferenced` 管理；应用通过 `RemoteRknnSession` 提交输入。设备侧不安装 Torch、Torchvision、ONNX Runtime、ONNX 或 RKNN-Toolkit2，也不调用本地 `rknnlite`。

## 第一个安装包可使用的现有模型

`model.example.json` 对应平台已经使用的 YOLOv8n COCO-80 模型：

| 项目 | 值 |
| --- | --- |
| 应用内 model_id | `yolo8n/1` |
| 文件 | `yolo8n_rawhead_int8.rknn` |
| 大小 | 4,339,284 字节 |
| SHA-256 | `5379721bb9b16fa3b9caceb8958175b2e8b0665d4ebcfe4e2524ad1db80d30da` |
| 编译目标 | RV1126B，RKNN-Toolkit2 2.3.2 |
| 设备运行时 | RKNN Runtime 2.3.2 |
| 输入 | RGB、uint8、NHWC、`[1,640,640,3]` |
| 输出 | 6 个 float32 张量，三个尺度分别为 80、40、20 |
| 解码 | DFL `reg_max=16`，类别分支为 **logits**，需要 sigmoid |

每个尺度先输出 `[1,64,H,H]` 的 box 分布，再输出 `[1,80,H,H]` 的类别分支。该 SHA 对应的真实模型有 **6 个输出，没有 score_sum 辅助输出**。输出共 4,838,400 字节。模型内部 INT8 量化与接口 float32 输出不矛盾：平台运行时会返回解量化后的 float32 张量。

上述信息来自相邻 SDK/ext-api 仓的 `apps/yolo-detector/manifest.json`，以及 SDK 项目既有的 `artifacts/inferenced-memory-investigation-20260909/synced-b3da397/device_backend_comparison.log` 中模型 SHA、`runtime_ready.description` 和 6 个 `output_comparison` 记录。该记录还包含 RKNN-Toolkit2/Runtime 2.3.2 和 `target platform: rv1126b`。这份 metadata 没有代表本应用已经完成新的真机精度测试。

构建者需要提供**完整且哈希一致的真实模型文件**。本仓示例 JSON 不附带模型权重，也不下载模型。已授权的设备模型路径是 `/userdata/local/apps/yolo-detector/models/yolo8n_rawhead_int8.rknn`；另一个应用不能凭这条路径获得访问权限。要打包为本应用，应把同一模型内容随本应用一起安装，并生成本应用自己的 manifest artifact 声明。

建议打包目录：

```text
<app-root>/
  manifest.json
  models/
    yolo8n/
      1/
        model.json                 # 从 model.example.json 复制
        yolo8n_rawhead_int8.rknn   # 校验上述大小、SHA-256
```

`LocalModelStore` 的根目录设置为 `<app-root>/models`。构建器应在写入安装包前调用 `LocalModelStore(root).get("yolo8n/1")`；它会读取模型文件并验证 SHA-256。不要通过改 metadata 中的 SHA 来接受未知替代模型：其他转换产物可能具有不同的输出顺序、类别顺序、sigmoid 或预处理契约。

## metadata 和平台权限是两个层次

`model.json` 描述图像、张量和后处理。它不授予 NPU 权限。安装包 manifest 必须同时具备：

- `permissions.sdk` 包含 `npu.infer`；`resources.claims` 声明 `npu.rknn` 的 `scheduled` 模式。
- `artifacts` 中有 `kind: rknn`、`source: bundled`，并写入真实 `file`、`sha256` 和 `size`。文件相对路径指向本应用包内的模型。
- `models` 中的 `file` 与 artifact 相同，`input` 为 `[1,640,640,3]`；保持与 metadata 输入完全一致。
- 平台兼容声明匹配 RV1126B、AArch64、CPython 3.11 和 RKNN Runtime 2.3.2。

当前 appmgr 从 manifest 的 `models[].input` 生成可信输入：名称固定为 `input`，dtype 为 `uint8`，四维输入 layout 为 `NHWC`。后端提交的 `ModelSpec.inputs` 也使用 `input`；metadata 的 `input.name: images` 保存模型原始图输入名称。两者不是同一个命名空间。

当前 `inferenced` 以 appmgr 的可信声明重建模型输入协议，按顺序校验 shape/dtype；不会把客户端提供的 `ModelSpec`、内存预算、优先级或 SHA 当作授权策略。客户端仍在加载前核对本地 metadata/模型 SHA，并在每帧核对输出 shape/dtype。

appmgr 在启动时注入 socket、app ID、instance 和 generation。最终授权还绑定进程 PID 与其开始时间，并在操作时重新核查。手工启动可以运行健康检查和 metadata 查询，但模型加载返回 `npu_authorization_required`。复制环境变量、启动子进程、读取另一个应用的模型文件，都不能替代平台授权。应用也不会失败后回退到本地 RKNN 或 CPU 推理。

## 输入与颜色

HTTP 图片解码和原 Workflow 图像在应用内部使用 **BGR HWC uint8**。`EdgeModelManager.infer()` 和 `infer_from_request_sync()` 的 NumPy 输入均遵循这一约定。

`KitBackend` 只在一个地方转换 BGR → RGB，再使用 Kit 的 letterbox 预处理，返回 **RGB NHWC uint8** 输入。可选的 `input.padding_value` 为 0–255 的整数，默认 114；Roboflow 模型包使用黑色补边时自动写入 0。模型归一化必须已在转换时烘焙，本应用不会再次除以 255。metadata 中 `input.color_format` 必须为 `RGB`，`normalization` 必须为 `baked`。

物理 NPU batch 固定为 1。manager 可以接收最多 8 张图片的列表，并逐张串行执行。应用只保持一个驻留 session，所有 load/infer/release 由同一把锁串行化。释放失败时保留句柄，拒绝继续推理及装入其他模型，允许重试释放。

HTTP/Workflow 预测框为原图像素坐标，`x/y` 是中心，另有 `width/height/class/class_id/confidence`。向平台 OSD/result sink 回注时需要由结果输出层按帧尺寸归一化为 `[0,1]`；不得把 HTTP 像素框直接当归一化坐标发送。

## 支持的输出协议

所有输出必须在 `outputs` 中按**运行时实际顺序**声明 `name/shape/layout/dtype`。dtype 目前仅支持运行时 float32，形状为固定正整数，总输出上限 64 MiB。输入仅支持 batch=1、三通道、固定正方形、边长不超过 1280 的 NHWC uint8。

### `yolo-decoded`

单输出 `[1,C,候选数]`（`layout: BCN`）或 `[1,候选数,C]`（`BNC`）。候选数要求大于 C。前四个通道为网络输入像素空间的 `cx/cy/w/h`，默认 C 为 `4+类别数`。

```json
{"kind": "yolo-decoded", "box_format": "xywh", "scores": "probabilities"}
```

`scores` 也可显式设置为 `logits`；此时后端做 sigmoid，不能根据本帧值域猜测是否需要 sigmoid。单输出的 `role` 可省略或设为 `detections`。

v5/v7 带 objectness 的已解码导出应声明 `objectness: true`，C 为 `5+类别数`；
第五通道为 objectness，后续通道为类别分数。阈值应用于 objectness × 类别概率，
不能忽略第五通道。`scores: logits` 时两者分别 sigmoid 后再相乘。

### `yolo-end2end`

单个 BNC float32 `[1,N,6]`，每行是输入像素空间 `x1,y1,x2,y2,score,class_id`。
使用 `{"kind":"yolo-end2end","box_format":"xyxy","scores":"probabilities"}`。
类别 ID 必须为合法整数，置信度为 0–1。已经完成模型内筛选，禁止再次 NMS；
仍执行 Workflow 的置信度、类别过滤、坐标还原和最大结果数限制。

### `yolo-distance`

每尺度两个 NCHW 输出，显式 `role: boxes` 为 `[1,4,H,H]` 的 LTRB 距离，
`role: scores` 为 `[1,类别数,H,H]`。中心点为 `(grid_x+0.5, grid_y+0.5)`，
距离以当前网格步长为单位；这里没有 DFL softmax。

YOLO26 one2one 原始头使用
`{"kind":"yolo-distance","scores":"logits","nms":false,"topk":300}`。
one2many 分支使用 `nms:true`。两种分支不能互相替代。

对于 `yolo-distance` 或 `yolo-dfl` 的 `nms:false` 原始头，先按最大类别分数
选 Top-K 候选，再在候选的全部类别分数中选 Top-K；保留同一候选的多个类别。
`topk` 来自源端到端输出长度，范围 1–1000，旧 metadata 默认 300；筛选先于
Workflow 类别过滤。NMS 模式不接受此参数。

### `yolo-dfl`

每尺度两个 NCHW 输出，`role: boxes` 为 `[1,64,H,H]`，`role: scores` 为 `[1,类别数,H,H]`。每个 H 必须整除输入边长，`reg_max` 固定为 16。

```json
{"kind": "yolo-dfl", "reg_max": 16, "scores": "logits"}
```

类别分支可以声明为 `probabilities` 或 `logits`；前者逐帧校验 `[0,1]`，后者显式应用稳定 sigmoid。box 分支为分布 logits，交给 Kit DFL 解码。标准 NMS 使用 `kit.runtime.postprocess.detect.nms` 的 IoU 语义，支持按类或 class-agnostic、类别过滤、候选上限与结果数上限。

兼容六输出包时，可以省略 `role` 并由明确的 64/类别数通道识别 box/class。新包建议始终声明 `role`。类别数为 64 暂不支持，因为当前 Kit DFL 解码器按通道区分分支，无法消除歧义。

部分导出器会在每尺度附加 `[1,1,H,H]` 的辅助分数和。仅当 metadata 显式标记 `role: score_sum` 时允许：运行时仍检查张量 shape/dtype/有限值，但此辅助值不参与解码，也不作为 objectness 相乘。每尺度最多一个该辅助输出。未声明用途的额外输出会被拒绝；不能把语义不同的 objectness 输出标为 `score_sum`。

当前 Kit DFL 复用路径包含其私有 `_decode_dfl` helper，需与经验证的固件 Kit 版本一起验收。升级 Kit 后应重跑输出/预后处理测试和真机精度对比。

## 当前不支持

- 老 RV1126 格式的 RKNN；RV1126B 必须使用对应 RKNN-Toolkit2 目标。
- `.pt`、TorchScript、直接 `.onnx`、TensorRT，以及设备端训练/转换。
- 未定义输出协议的 YOLO、缺少锚点定义的原始 anchor 检测头、多张量 NMS 输出、分类、分割、关键点、OCR 和大模型。
- 多输入、动态 shape、NPU batch>1、非正方形输入、float32/NCHW 模型输入、自定义归一化、未声明的辅助头。
- 以未校准的转换产物替代现有模型后直接假定精度不变。

支持新模型应先确认真实张量、预处理和后处理语义，再扩展 metadata/decoder 并补针对性测试。不能仅改文件扩展名。

## 后续远端转换

应用中心已接入 Roboflow 下载与 SenseCraft 转换，见
[Workflow 模型准备](../../docs/rv1126b_workflow_models.md)。下述接口是另一条保留的
自定义转换扩展，不是应用中心自动转换流程。

`inference.edge.conversion` 定义了可注入的 `Converter`、`ConversionRequest`、`ConversionResult` 和 `ConversionService`。没有配置提供方时，状态为 `unconfigured`，转换请求明确返回 `converter_unconfigured`。当前没有假设任何 HTTP 地址、鉴权头、任务 JSON 或下载接口。

后续转换服务需要约定源模型引用、RV1126B 目标、工具链/运行时版本、量化与校准数据、类别顺序、输入及输出契约、产物校验和、任务失败与重试语义。转换完成后仍需经过模型包校验、应用 artifact 更新与 appmgr 授权；转换结果不会自动覆盖设备模型资产。
