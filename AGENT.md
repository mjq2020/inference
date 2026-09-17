# ARM64 部署记录与注意事项

**历史记录说明（2026-09-15）：** 下文描述早期完整 CPU 环境。当前 RV1126B 应用已改为 `inference-rv1126b==0.3.1`，使用原生 Workflow 画布与引擎、RKNN/NPU 推理，不安装 Torch、ORT 或训练模块。当前构建、启动配置和使用方式请见 [RV1126B 部署说明](deploy/rv1126b/README.md) 与 [Workflow 说明](docs/rv1126b_workflows.md)，不要按下文重新安装完整 CPU 依赖来升级当前应用。

本文记录本项目在 ARM64 嵌入式设备上运行 HTTP 服务、Workflows 和视频流管理接口的部署方法，供后续部署和维护参考。仓库开发规范仍见 [AGENTS.md](AGENTS.md)。设备地址、密码、API Key 和 SSH 主机密钥不写入本文；命令中的 `<DEVICE_IP>` 由操作者替换。

记录日期：2026-09-14。以下区分设备已验证的行为、重建流程和待验证的优化，不应将单台设备的结果推广为所有 ARM64 平台的保证。

## 1. 已验证的平台与能力

| 项目 | 本次部署配置 |
|---|---|
| 设备 | Rockchip RV1126B reCamera Pro，`aarch64` |
| 系统 | Buildroot 2023.02.6，glibc 2.38 |
| Python | 设备 CPython 3.11.6，独立 venv |
| 内存 | 约 2 GB，无 swap |
| 应用 | `inference[http]==1.5.2`、`inference-models==0.37.0` |
| 推理依赖 | PyTorch `2.14.0+cpu`、TorchVision `0.29.0+cpu`、ONNX Runtime `1.21.1` |
| 图像依赖 | NumPy `2.3.5`，OpenCV 分发包版本 `4.11.0.86` |
| 部署方式 | 开发机准备 ARM64 依赖包，设备原生运行，无需设备安装 Docker |
| 持久化目录 | `/userdata/roboflow-inference` |
| HTTP 端口 | `9001` |

本次安装以发布的 Python 包为基础，补充当前仓库提交 `3e69d06b2` 的 Builder HTML 和 landing 静态资源。它不是通过在设备上执行 `pip install -e .` 安装整个开发环境。仓库根目录的 `setup.py` 会带入开发、测试和多种模型依赖，不宜直接作为小内存设备的安装入口。

HTTP 服务和 Workflow 执行引擎可用，不表示所有可选模型都已安装或能在这台设备上运行。不同节点仍可能需要额外依赖、模型权重、凭据或网络。模型和视频负载需要分别验收。

当前本地模型使用 CPU 后端。设备有 RK NPU，不会使 ONNX Runtime 的 `CPUExecutionProvider` 或 PyTorch CPU 自动使用它。接入 RV1126B NPU 需要匹配的 RKNN 模型、驱动和运行时，以及模型前后处理和 Inference/Workflow 适配。本次部署未实现或测试 NPU 推理。调用远端模型服务的节点，其模型计算在所调用的远端服务执行。

## 2. 部署前检查

在设备上确认架构、Python、libc、内存和持久分区容量：

```sh
uname -m
cat /etc/os-release
python3 --version
python3 -c 'import os; print(os.confstr("CS_GNU_LIBC_VERSION"))'
df -h /userdata
cat /proc/meminfo
```

安装包只验证了上述 ARM64、CPython 3.11、glibc 固件组合。32 位 ARM、musl 系统或其他 Python 次版本需要重新准备依赖；不能直接复制 x86_64 的 venv 或二进制 wheel。当前 `inference-models` 的 Python 要求为 `>=3.10,<3.13`，不能仅依据根目录开发说明中的最低 Python 版本选择运行环境。

完整安装至少预留约 3 GB 可用存储，并另外预留模型缓存、日志和更新空间。以解包后的实际占用为准；压缩包大小不能代表所需容量。检查原有相机服务的内存占用，2 GB 设备尤其需要验证加载模型后的峰值。

通过 SSH 部署时保留主机密钥校验。刷固件可能使主机密钥改变，应核实设备身份和新指纹后更新对应记录，不能用关闭校验代替核实。密码使用交互输入或既有的 SSH 密钥，不写入命令、脚本或版本库。

## 3. 部署产物与重建方法

本次完整安装包名称为 `inference-1.5.2-recamerapro-arm64.tar.gz`，附带同名 `.sha256` 校验文件。安装包和构建辅助脚本是单独保存的部署产物，**没有随本文加入 Git 仓库**。复用以下流程前，需要取得这些产物；只有代码仓库和本文不足以重建完全相同的依赖集合。

| 产物 | 用途 |
|---|---|
| `requirements.txt`、`pylock.toml` | 完整依赖版本与来源；重建时保留，避免重新解析为其他版本 |
| `build.sh` | 在 ARM64 Python 构建环境中安装依赖 |
| `collect_libraries.py` | 收集独立运行时动态库，排除系统 glibc 和动态加载器 |
| `collect_stdlib.py` | 补充固件裁剪掉的 Python 标准扩展模块 |
| `bundle/site-packages/`、`bundle/lib/` | 构建出的 Python 依赖和动态库 |
| `server.py`、`server.env`、`run.sh`、`service.sh` | 服务入口、配置、环境设置和进程管理 |
| `activate.sh`、`finish_install.sh` | 交互使用环境及安装后的 venv/命令入口初始化 |
| `S95roboflow-inference`、`install_autostart.sh` | Buildroot 自启动入口及恢复工具 |
| `smoke_test.py`、`crop_workflow.json`、`example_workflow.py` | HTTP 和实际 Workflow 验收 |
| `package_bundle.py` | 打包上述运行时文件并生成 SHA-256 |

重新准备依赖时，使用原生 ARM64 开发机，或已配置 ARM64 执行能力的开发机容器。此次构建使用 ARM64 `python:3.11-bookworm` 环境，Python 3.11.16、glibc 2.36；跨架构容器通过 QEMU 执行。更换构建镜像后应记录镜像 digest，并重新验证设备兼容性。

构建按以下顺序进行：

1. 将单独保存的构建脚本和锁定依赖放入构建工作目录，在 ARM64 容器中将该目录挂载到 `/work`。
2. 执行 `build.sh`。脚本安装 `libgl1`、`libglib2.0-0`、`libvips42`、`libgomp1`，并通过 `uv pip install --torch-backend cpu` 将锁定依赖安装到 `/work/bundle/site-packages`。设置 `SAM2_BUILD_CUDA=0`，避免引入 CUDA 构建路径。
3. 执行 `collect_libraries.py`，补齐 OpenCV、pyvips、压缩库等的传递依赖。对缺失库和符号版本错误逐项检查，不替换设备的 glibc、动态加载器或系统 Python。
4. 执行 `collect_stdlib.py`，补充 `_bz2`、`_lzma`、`_sqlite3`、`_uuid`、`_decimal` 和 `sqlite3` 包。它们放入独立运行环境，不写入系统标准库目录。
5. 将 pip 安装进依赖目录；本次环境另包含 `pip==26.2.1`。`finish_install.sh` 使用 `venv --without-pip`，因此不能假定设备固件有 `ensurepip`。
6. 补齐发布 wheel 遗漏的 Builder HTML，以及工作目录下 `inference/landing/out` 的静态资源。资源应与应用版本匹配，不能仅验证 Python import。
7. 在构建环境检查导入和依赖一致性，再执行 `package_bundle.py` 生成安装包及校验文件。最终兼容性以设备上的验收为准。

补充 CPython 扩展需要匹配架构、Python ABI 和共享库要求。本次从 CPython 3.11.16 补充的扩展已在设备 3.11.6 上通过基础测试，不代表任意跨补丁版本、构建选项的二进制都兼容。

本次依赖同时包含 `opencv-python`、`opencv-contrib-python`、`opencv-python-headless`，均固定为 `4.11.0.86`。这些包共享 `cv2` 命名空间；不要单独升级其中一个，重建或调整组合后须重新验证图像功能和依赖关系。这是现有部署记录，不是推荐所有新环境同时安装三者。

## 4. 安装到兼容设备

以下示例在开发机 Bash 中运行，当前目录包含可信来源的安装包及校验文件。它用于全新安装；目标目录已存在时会停止，避免覆盖现有配置和数据。

```bash
set -euo pipefail
DEVICE='root@<DEVICE_IP>'
DEPLOY_ARCHIVE='inference-1.5.2-recamerapro-arm64.tar.gz'

sha256sum -c "${DEPLOY_ARCHIVE}.sha256"
ssh "$DEVICE" 'test ! -e /userdata/roboflow-inference && mkdir -p /userdata/roboflow-inference'
gzip -dc "$DEPLOY_ARCHIVE" | ssh "$DEVICE" 'tar -xf - -C /userdata/roboflow-inference'
ssh "$DEVICE" 'sh /userdata/roboflow-inference/finish_install.sh'
```

本次固件的 BusyBox tar 不支持 `-z`，所以在开发机解压 gzip 层，通过 SSH 传送 tar 数据。设备不必同时存放压缩包和解包后的运行时。任一步失败应先检查部分安装目录，再处理重试，不要继续启动不完整的安装。

已有安装需要更新时，先备份 `server.env`、启动脚本、模型缓存和实际使用的 Workflow 定义/应用数据，保留旧运行时用于回退。在维护时段停止服务后再切换文件，不能直接向运行中的 venv 覆盖依赖。Workflow 定义可能保存在客户端或云端，应按实际保存位置备份。

最终运行时、共享库和缓存均应留在持久分区。不要保留指向临时构建目录或易被刷机覆盖的 `/opt` 目录的依赖软链接；venv 的解释器入口仍需兼容当前固件的系统 Python。

## 5. 运行配置与进程管理

本次 `server.env` 的关键配置如下。该文件由 shell 加载，必须保持合法的 shell 赋值语法：

```sh
HOST=0.0.0.0
PORT=9001
NUM_WORKERS=1
MODEL_CACHE_DIR=/userdata/roboflow-inference/cache
INFERENCE_HOME=/userdata/roboflow-inference/cache
MAX_ACTIVE_MODELS=1
WORKFLOWS_STEP_EXECUTION_MODE=local
WORKFLOWS_MAX_CONCURRENT_STEPS=1
ENABLE_STREAM_API=True
STREAM_API_PRELOADED_PROCESSES=0
USE_INFERENCE_MODELS=True
ONNXRUNTIME_EXECUTION_PROVIDERS='[CPUExecutionProvider]'
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
PYTHONDONTWRITEBYTECODE=1
```

`run.sh` 切换到安装目录、加载配置、仅为当前进程设置 `LD_LIBRARY_PATH`，并将日志写入 `logs/server.log`。不要把私有动态库搜索路径全局写入系统环境，以免影响相机程序和系统自带的 NumPy、OpenCV、RKNN 包。

`server.py` 以一个 Uvicorn worker 运行 `HttpInterface`，并管理视频流子进程。仅设置 `ENABLE_STREAM_API=True` 不会替代管理进程的实际启动。`STREAM_API_PRELOADED_PROCESSES=0` 表示不预热 Pipeline 工作进程，不表示视频管理进程不存在。`MAX_ACTIVE_MODELS=1` 是此启动器传给模型缓存管理器的限制，不是整个设备的全局内存上限。

为减少 Linux fork 后的内存复制，现有启动器在大量导入前执行 `gc.disable()`，构建 HTTP 接口后执行 `gc.collect()`、`gc.freeze()`，再创建子进程，并分别恢复父子进程的 GC。子进程设置父进程退出信号，主进程退出时也负责终止并回收子进程。修改启动器时须保留这些行为，并验证停止服务后没有遗留进程。该处理依赖 Linux/fork；不能直接视作其他操作系统或 multiprocessing 启动方式的通用方案。

在设备上管理服务：

```sh
/userdata/roboflow-inference/service.sh start
/userdata/roboflow-inference/service.sh status
tail -n 100 /userdata/roboflow-inference/logs/server.log
```

修改配置后可执行 `service.sh restart`，停止服务使用 `service.sh stop`。`status` 只表示进程存活；`/healthz` 返回 HTTP 200 才表示接口就绪。本次完整服务启动需要约 4–5 分钟，不能因几十秒内端口未打开就反复启动或重启。

`HOST=0.0.0.0` 会监听设备所有网络接口，应在预期的受控网络内开放服务。需要私有模型或托管 Workflow 时再提供对应的 API Key，存放凭据的配置文件应限制读取权限。Roboflow API Key 用于相关资源访问，不能据此认为所有本地 HTTP 端点已具备访问认证。

## 6. 验收与访问

浏览器访问 `http://<DEVICE_IP>:9001/docs` 查看 API，访问 `http://<DEVICE_IP>:9001/build` 打开 Builder 入口。当前 Builder 通过 iframe 加载 Roboflow 在线编辑器，浏览器需要联网。入口返回 200 不等于浏览器编辑、保存等交互均已验证。

在设备上运行安装包附带的验收脚本：

```sh
. /userdata/roboflow-inference/activate.sh
python3 -m pip check
python3 /userdata/roboflow-inference/smoke_test.py
python3 /userdata/roboflow-inference/example_workflow.py
```

`smoke_test.py` 检查 `/healthz`、`/docs`、`/openapi.json`、`/build`、`/workflows/execution_engine/versions`、`/inference_pipelines/list`，并读取 `/workflows/blocks/describe`。本次版本返回 236 个块描述；版本变化时不要把这个数字作为永久固定要求。

脚本还通过 `POST /workflows/run` 将 64×48 图像裁剪为 32×24，验证真实 Workflow 执行。`example_workflow.py` 使用 SDK 执行同一裁剪流程。两者无需模型或 API Key；它们不能代替神经网络模型、摄像头流或 NPU 的验收。

通过 HTTP/SDK 提交本地 Workflow 定义不必打开在线 Builder；执行是否需要网络取决于所选节点、模型是否已缓存和外部资源。实际部署还应使用目标模型、真实分辨率和预期并发持续运行，检查输出正确性、延迟、内存峰值及停止后的资源释放。

## 7. 自启动与刷固件恢复

本次设备使用 Buildroot init 脚本，不使用 systemd。服务验收通过后，在设备上安装启动入口：

```sh
sh /userdata/roboflow-inference/install_autostart.sh
```

该脚本将持久分区内的 `S95roboflow-inference` 安装到 `/etc/init.d/S95roboflow-inference`，不会立即启动服务。正常重启验证应检查 `logs/autostart.log` 的本次 boot ID、开机时间和动作记录，再等待健康检查成功；不要先手动 start，再据此认定自启动有效。

2026-09-14 已实际验证：普通重启后启动入口约在系统开机 22 秒执行，发出重启命令后约 260 秒 HTTP 就绪，期间没有手动启动服务。这是该固件下的一次观测，不能作为所有设备的启动时限。

**刷固件与普通重启不同。** 已遇到 userdata 保留、但 `/etc/init.d/S95roboflow-inference` 被固件覆盖丢失的情况。刷入兼容固件后，先检查运行时、Python 和依赖是否完整，再恢复入口并启动：

```sh
sh /userdata/roboflow-inference/install_autostart.sh
/etc/init.d/S95roboflow-inference start
tail -n 20 /userdata/roboflow-inference/logs/autostart.log
tail -n 100 /userdata/roboflow-inference/logs/server.log
```

如果 userdata 也被清空，需要完整重装。若希望刷固件后自带启动入口，应将该 init 脚本纳入对应固件镜像。其他 ARM64 Linux 发行版应按其 init 系统配置服务，不能直接照搬 Buildroot 的启动路径。

## 8. 内存与故障排查

2026-09-14 原生设备测量：未加载模型、没有视频 Pipeline 时，HTTP 进程 PSS 约 802.5 MiB，视频管理进程约 590.2 MiB，合计约 1392.7 MiB，系统 `MemAvailable` 约 331.2 MiB。约一分钟采样内占用稳定，但不足以排除长期运行或实际模型负载下的泄漏。

多进程共享页面会被 RSS 重复计算，评估整套服务应汇总 PSS。可从 `/proc/<PID>/smaps_rollup` 读取 PSS，从 `/proc/meminfo` 查看可用内存。模型缓存目录的磁盘大小不等于 RAM 占用；整机 load average 高也不一定表示 Inference 占满 CPU，应同时检查进程 CPU 和 D 状态线程。

高基线主要来自提前加载 PyTorch/TorchVision 等依赖、大量模型与 Workflow 模块、块描述与 Schema、视频管理子进程的写时复制，以及分配器保留的空闲堆内存。当前已限制 worker、步骤并发、活动模型缓存和计算线程数，但这些限制不能消除导入开销。仅设置 `WORKFLOW_DISABLED_BLOCK_TYPES/PATTERNS` 会在导入之后过滤节点，不会避免前面的内存分配。

**以下是优化候选，没有应用到本次设备运行时：** 在启动时 `gc.collect()` 之后、`gc.freeze()`/fork 之前调用 glibc `malloc_trim(0)`，在开发机的 ARM64/QEMU 隔离实验中使完整配置单进程 RSS 减少约 372 MiB。健康检查、OpenAPI 和块描述接口通过，但完整候选启动器的设备多进程生命周期、真实模型和视频负载尚未验证。不能将这个数值当作设备已经节省的内存，也不应机械地在每个请求后调用。

进一步优化可考虑模型/节点按需导入和视频管理进程按需启动。关闭部分可选模型在隔离实验中降低约 181 MiB，但会减少能力；关闭 `ENABLE_STREAM_API` 会取消视频 Pipeline 接口。需要完整视频功能时应保留它。不同优化的收益可能重叠，不能简单相加。

| 现象 | 优先检查 |
|---|---|
| `_bz2`、`_sqlite3` 等导入失败 | 固件标准库裁剪、补充扩展 ABI 及动态库是否齐全 |
| `libGL.so`、`libvips.so` 等缺失 | 私有 `lib/` 和 `run.sh`/`activate.sh` 的库搜索路径 |
| `GLIBC_* not found`、错误的 ELF 架构 | 构建环境与设备的 libc/架构兼容性；重新构建匹配依赖 |
| Builder 或静态资源 404 | HTML 是否打包、landing 资源是否齐全、服务工作目录是否正确 |
| 进程存活但 HTTP 不通 | 启动日志、导入是否仍在进行、监听端口及绑定地址 |
| 列出 Workflow 块或加载模型时被杀 | 内核 OOM 日志、可用内存、并发和 fork 前的 GC 处理 |
| 停止后内存未释放 | 是否残留视频管理/工作进程，以及是否只是文件缓存 |
| 普通重启后未启动 | init 入口权限、userdata 挂载顺序、本次 `autostart.log` |
| 刷固件后未启动 | 系统分区入口是否丢失、userdata 是否保留、固件 ABI 是否变化 |
| 模型没有使用 RK NPU | 当前后端仍为 CPU，是否已单独实现并验证 RKNN 适配 |

优先使用日志、进程信息和实际请求定位问题，不通过删除模型缓存、清空系统 page cache 或修改系统 Python 来处理框架的常驻内存开销。
