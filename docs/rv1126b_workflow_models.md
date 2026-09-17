# RV1126B Workflow 模型准备

Inference 应用 0.3.5 支持保存工作流后自动下载 Roboflow 模型、在线转换和绑定，
并扩展了 YOLO 导出格式识别和后处理。
需要同时部署 appmgr 的 `workflow_models.py`、`workflow_roboflow.py`、
`workflow_onnx.py`、`workflow_cloud.py`、`workflow_model_contract.py`，以及
服务路由、启动依赖检查、模型授权代码和原生 Web 前端。

## 默认使用流程

1. 在原 Roboflow 画布中选择模型并保存工作流。模型列表提供
   `yolov8n-640`、`yolov11n-640` 和 `yolo26n-640` 公开预训练模型入口，
   不要求提前导入 RKNN。
2. 系统从保存的工作流读取 `model_id`，解析 Roboflow 模型包，下载 ONNX、
   类别和推理配置。开放预训练模型无需 Roboflow API Key。需要授权的模型
   使用应用已有的可选 Roboflow API Key；被拒绝时明确提示，不绕过权限。
3. 已登录 SenseCraft 的设备自动转换。没有登录时，应用卡片显示登录提示，
   “工作流模型”内提供设备原有的 OAuth 授权链接。也可在内置模型管理中登录；
   后台自动继续，无需重新提交或保持页面打开。
4. 系统下载 RKNN，读取转换后实际输入输出描述，自动生成设备配置并绑定原
   `model_id`。SenseCraft 会将部分 YOLO 模型的解码输出改为六个原始检测分支，
   系统按实际产物适配，保留模型原本的类别顺序和补边颜色。
5. 运行中的应用通过正常生命周期短暂重启，更新精确 NPU 授权，并做一次真实
   NPU 推理验证。应用停止时，下一次启动自动验证。画布会话在有效期内跨重启
   保留；退出登录或修改设备访问口令仍会撤销相应授权。
6. 模型尚未准备完成时点击启动，应用保留启动请求，等待依赖就绪后自动运行。
   点击停止会取消启动意图。模型准备任务本身可单独取消。

应用卡片显示当前准备阶段；“工作流模型”按模型合并状态，避免同时列出模型和
重复的转换任务。正常流程不要求填写模型 ID、model.json 或云端任务 ID。

## 上传自己的模型

“工作流模型 → 上传模型”默认只需一个 ONNX 文件。支持带 Ultralytics 模型及
类别元数据、固定 batch 1 和固定方形输入的 YOLO 检测导出（范围见下表）。
系统自动读取类别、输入输出并生成 ID；有多个缺失模型时可选择绑定对象。
缺失类别、动态尺寸、未知结构或不支持的预处理会明确报错，不猜测配置。

“高级导入”保留 ONNX + model.json、RKNN + model.json，以及 SenseCraft
已有产物导入。RKNN 本身通常不包含类别名称，因此任意 RKNN 裸文件不承诺
完全自动识别。手动配置格式见 [MODEL_FORMAT.md](../deploy/rv1126b/MODEL_FORMAT.md)。

## 支持范围与轻量实现

自动准备流程识别以下架构及输出契约。架构名称不代表任意导出文件都兼容，
还需满足 RGB、固定尺寸、letterbox、mean 0 / std 255 和下列输出要求。

| 架构 | 接入的检测输出 |
| --- | --- |
| YOLOv5、YOLOv7 | 已解码 xywh + objectness + 类别概率，最终置信度为两者相乘 |
| YOLOv5u、YOLOv8、YOLOv9、YOLO11、YOLO12 | 已解码 xywh + 类别概率；转换器保留的标准 DFL 分支 |
| YOLOv10、YOLO26 | `[1,N,6]` 端到端 xyxy/置信度/类别；已识别的 one2one 原始头 |
| YOLO26 原始头 | 四通道 LTRB 距离分支 + 类别分支；按源输出 Top-K 筛选，不重复 NMS |

YOLO26 和 YOLOv8 已完成公开模型下载、SenseCraft 转换、真实图片 NPU 推理。
其余上述导出格式通过自动化契约与解码测试，尚未逐一完成真机精度验收。
分类、分割、姿态、YOLOX，以及缺少锚点信息的 v5/v7 原始检测头仍需单独适配。
下载权限、转换服务支持的算子和设备内存预算也会影响具体模型是否可用。

- 输入为 batch 1、方形、RGB/NHWC uint8。支持 `yolo-dfl`、`yolo-decoded`、
  `yolo-end2end`、`yolo-distance`；预处理和输出配置不能互换。
- 自动读取 RKNN v6 的有界编译描述；不识别的产物拒绝自动绑定。
  编译描述仅用于生成配置，最终仍由真实 NPU 加载及输出检查验证。
- 下载复用官方 Inference 的模型包提供机制。API Key 仅发给 Roboflow API，
  不转发给模型下载地址，不写入任务状态或返回前端。下载按校验和验证。
- 同一账号、相同模型权重和转换输入条件的成功转换可复用，避免重复提交。
- 设备不安装 PyTorch、ONNX Runtime、ONNX Python 包或完整 RKNN Toolkit。
  ONNX 描述通过 mmap 有界读取，权重流式传输；转换在 SenseCraft 运行。
- 单文件最多 256 MiB，额外模型最多 4 个，另受磁盘和 NPU 内存预算限制。
  文件存储在 `/userdata`，使用 64 KiB 网络缓冲。

## 任务、权限和恢复

模型和任务保存在 `/userdata/local/appmgr/workflow-models/<app_id>/`，应用升级
不会覆盖。SenseCraft 凭据由系统 OAuth 会话持有和刷新。HTTPS 使用系统 CA。
模型路径、摘要和输入契约由 appmgr 按原有进程身份签发，保持 broker 授权边界。

工作流扫描、下载、转换轮询和登录恢复均在系统后台运行。网络下载失败有有界重试；
取消的任务不会因下一次扫描被重新创建。转换提交响应丢失时不自动重复提交，
通过已有转换记录恢复。删除模型必须先停止应用并解除保存工作流的引用。

## 管理 API

接口沿用设备 Web JWT 和同源写入限制。基础路径：
`/api/app-center/v1/apps/<app_id>/workflow-models`。

| 接口 | 用途 |
| --- | --- |
| `GET /` | 模型、任务和工作流依赖状态 |
| `POST /tasks`，`{mode:"roboflow", model_id:"yolov8n-640"}` | 显式准备 Roboflow 模型；保存工作流会自动创建 |
| `POST /tasks`，`{mode:"auto", filename:"my-model.onnx"}` | 单文件上传，无需 metadata；可选 model_id |
| `POST /tasks`，mode 为 onnx/rknn/cloud | 高级导入，需 metadata |
| `POST /tasks/<id>/source` | application/octet-stream 流式上传 |
| `POST /tasks/<id>/dataset` | 高级导入的可选校准 ZIP |
| `POST /tasks/<id>/resume` | 文件就绪或失败后重试 |
| `POST /tasks/<id>/cancel` | 取消本地准备并清理暂存文件 |
| `GET /cloud-records` | 当前账号的转换记录 |
| `POST /remove` | 移除无引用的额外模型 |

SenseCraft 登录缺失返回 `409/sensecraft_login_required`，不触发设备 Web 会话
的 401 注销。原 inference `/model-conversion` 扩展接口仍保留给独立转换协议。
