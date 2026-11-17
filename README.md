# 双副本 GPU 推理预实验

| 切片 | 交付与验收 |
|---|---|
| M0 | Python 3.11+ 标准库的 CPU 合成回放、影子分析和配对。合成样本仅验算程序，研究结论留空。 |
| M1a | Kubernetes 清单和真机配置模板、分项预检、采集与核验接口、mock 独占及清理演练。K01–K07 的现场状态均为 `not_tested`。 |
| M1b | 真实动作回放接口接收冻结计划分支。真实执行入口返回 `BLOCKED_UNTIL_HARDWARE`，报告「真实执行后端尚未接入」。 |
| M1c | 设备采集接口保留采样、发布时间、时钟映射、标记记录和查询结果。真实采集入口返回 `BLOCKED_UNTIL_HARDWARE`。 |
| M2 | `LiveReplay` 接口预留两组策略各自的独立闭环回合、真实触发及派发时间、全池与 B 既有群计分。 |

## 目录结构

```text
replaylab/
├── replaylab/             # Python 包:入口与全部实现
│   ├── __main__.py        # python -m replaylab
│   ├── cli.py             # 四个子命令与退出码
│   ├── preflight.py       # M1a 分项预检
│   ├── hardware.py        # 台架契约、Lease、注入器、清单渲染
│   ├── live.py            # M2 闭环接口
│   └── ...                # 契约、路由、影子、配对、统计等
├── tests/                 # 自动化验收(T01–T15、K04/K05 mock)
├── fixtures/
│   ├── synthetic/         # M0 合成输入与期望摘要
│   └── hardware/          # 真机配置模板与 mock 预检
├── deploy/kubernetes/     # 可渲染的 Kubernetes 清单模板
├── scripts/verify.ps1     # 一键运行全量测试
├── out/                   # 运行产物(可重新生成)
├── pyproject.toml
├── requirements.txt
└── README.md
```

本目录即工程根目录。无需安装,直接执行 `python -m replaylab`。

## 环境

- Python 3.11+(当前验证于 3.14)。
- M0/M1a 只使用标准库,`requirements.txt` 暂无第三方依赖。
- K8s 与真机适配依赖在接入 M1b/M1c 时单独锁定版本。

```powershell
python -m unittest discover -s tests -v
```

也可执行 `powershell -File scripts/verify.ps1`。

## M0 复现

以下命令都在本目录执行。每条命令使用新的输出目录。

```text
python -m replaylab preflight --config fixtures/synthetic/study.json --out out/m0/preflight
python -m replaylab run --config fixtures/synthetic/study.json --plan fixtures/synthetic/plan.json --out out/m0/run
python -m replaylab shadow --config fixtures/synthetic/study.json --manifest out/m0/run/manifest.json --out out/m0/shadow
python -m replaylab paired --config fixtures/synthetic/study.json --manifest out/m0/run/manifest.json --shadow out/m0/shadow/windows.csv --out out/m0/paired
python -m unittest discover -s tests -v
```

下表各筛选项加在 `python -m unittest discover -s tests -v` 后面。每项使用自动测试内的固定夹具。

| 用例 | 筛选项 | 验收证据 |
|---|---|---|
| T01 | `-k test_t01` | 两个独立进程、不同 `PYTHONHASHSEED` 下四条 CLI 文件清单和字节一致。核对三差值、A/A、日志引用和 manifest 原始摘要。 |
| T02 | `-k test_t02` | 同刻网关顺序、派发游标的 TTL、两组共享证据和账本缺口。 |
| T03 | `-k test_t03 -k test_t01` | 未核验原因、完成迟报、错代及迭代绑定。修改离线真值后复跑比对。 |
| T04 | `-k test_t04` | 到达即过期、已封存迭代和换代旧消息分开记原因。 |
| T05 | `-k test_t05` | 同刻启用、半开区间、决定与发送跨截止、定时退出。 |
| T06 | `-k test_t06` | 挡回、默认 B、发送后失败、剩余容量竞争、ACK 与状态快照。 |
| T07 | `-k test_t07` | G 为 40/50/42，差值 +10/+2/+8。A/A 独列，验证块数与合成研究样本数分开。 |
| T08 | `-k test_t08` | B 既有群与对照方向、发送晚于预定到达、截止等号和缺结局。 |
| T09 | `-k test_t09 -k test_t01` | 用实际影子产品查精确档、未测档及已安排但无有效结果的档位。 |
| T10 | `-k test_t10 -k test_artifacts -k test_paired_preserves` | 原始哈希、坏分支隔离、群和来源错配及 A/A 重复。 |
| T11 | `-k test_t11` | A/A 原始差、保留全窗、共同块分母、四格前置条件和 B 护栏。 |
| T12 | `-k test_t12` | 轻重工作量阈值、时效和时钟误差。 |
| T13 | `-k test_t13` | 共享公开进展与真实在途 OR 项同刻触发，领先为 0 ns。 |
| T14 | `-k test_t14 -k test_progress` | 事件与迭代去重、乱序和水位、同端点不同迭代、EMA 更新边界。 |
| T15 | `-k test_t15 -k test_real_record_contract -k test_hardware` | 分项状态、权限及台架缺项、mock 生命周期、引擎进展降级、动作配对和设备领先拆分。 |

## M1a 预检和部署

```text
python -m replaylab preflight --config fixtures/hardware/study.json --out out/m1a/declared
python -m replaylab preflight --config fixtures/hardware/study.json --mock-platform fixtures/hardware/mock_preflight.json --out out/m1a/mock
```

`preflight.json` 记录每项状态、来源、原因、具体缺项和证据引用。已执行的检查给 `pass` 或 `fail`。明确缺前置条件给 `blocked`。尚未执行的真实探测给 `not_tested`。两条命令也写 `status.json` 和 `summary.json`。未填写模板返回 3，mock 生成可读报告返回 0，结构错误返回 2。mock 报告的每一项均标记 `synthetic`。

`action_replay_status`、`inflight_timing_status` 和 `device_capability` 分别报告。仅能见到 `engine_progress_only` 时仍保留公共进展基线。平台、SLO 时间和配对证据完整的既有真机档案仍可离线计分，设备领先标为未验证。即使设备核验通过，每条在途证据也须独立核对可见性、时效和时钟误差。

### 准备清单

1. 复制 `fixtures/hardware/study.json` 到站点配置路径。按现场确认填写命名空间、节点对象名、该节点的 `kubernetes.io/hostname` 标签值、A/B 名称和镜像 digest、就绪探针及 PVC 参数。已授权的操作员身份填入 `recovery_subject`。其余未知项保留 `null`。`preflight` 和 `run` 分别列出本命令的缺项。
2. 固定实际模型与 tokenizer 的 revision、vLLM 版本及 build ID，保存镜像 digest 与构建的对应关系。真实接入前还要记录 Python、CUDA 运行时、NVIDIA 驱动和设备插件版本，按所选构建实测兼容性。A/B 使用同一后端镜像、推理参数与本地 KV。启动和就绪探针只检查服务与模型就绪。
3. 渲染器在本机生成 `bench.json` 和独立的 `lease-bootstrap.json`。四个工作负载用相同节点标签经调度器放置，A/B 分别申请一张 GPU。节点标签只约束位置，K01 在容器内实测物理 UUID 并核对授权。CPU 网关和 Job 使用同一 PVC，每次尝试写独立路径。网关镜像入口消费 `REPLAYLAB_A_SERVICE`、`REPLAYLAB_B_SERVICE`，回放镜像入口消费 `REPLAYLAB_GATEWAY_SERVICE`，流式工作负载走网关。Job 镜像接入时还须读取冻结研究、计划与输入轨迹。 [Kubernetes 节点调度](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#nodeselector)

```text
python -m replaylab.hardware render --config site-study.json --template-dir deploy/kubernetes --out out/m1a/manifests
```

### 部署与恢复

在确认目标上下文、命名空间及 GPU 授权后，由部署负责人执行以下步骤。以下 PowerShell 示例中的 `$benchContext` 和 `$benchNamespace` 由现场填入。

```powershell
kubectl --context $benchContext -n $benchNamespace apply --dry-run=server -f out/m1a/manifests/bench.json
kubectl --context $benchContext -n $benchNamespace create -f out/m1a/manifests/lease-bootstrap.json
kubectl --context $benchContext -n $benchNamespace apply -f out/m1a/manifests/bench.json
```

- Lease 使用创建请求初始化。若返回 `AlreadyExists`，读取指定 Lease 以核对现有持有者并保留状态。重部署使用 `bench.json`。Lease 初始化文件只用于创建。命名空间中的 runner 仅能读取和更新指定 Lease，A 仅能读取该 Lease，runner 和网关可读取本命名空间所需 Pod、Service 和 EndpointSlice。 [Kubernetes RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/#referring-to-resources)
- 挂起只作用于回放 Job。部署 A/B 会申请 GPU。Job 固定 `parallelism: 1`、`completions: 1`、`backoffLimit: 0`、`restartPolicy: Never`。程序重复启动仍须经过独占。真实执行接入且动作前置条件合格后，按冻结计划为每次启动使用新 Job 名和新尝试路径。 [Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/#handling-pod-and-container-failures)
- runner 从 Pod UID 和新 `runner_attempt_id` 建立持有者身份，使用当前 `resourceVersion` 和持有者做条件更新。遇 `409 Conflict` 时重读并重新判断。只有核验当前持有者、本轮绑定和前次清理后才重置或注入。A 侧以 `run_id` 和 `attempt_id` 幂等处理启动，自行在冻结硬时限停止本轮进程组。
- 正常释放前确认干扰进程已退出、测试请求及预留已清理、台架可复位。清理未确认则保持持锁并记录阻塞。陈旧持有者由配置的负责人凭旧 runner 已停止、干扰和遗留请求已清理、台架可复位的证据，按当前版本条件解除。Lease 经过时间本身不授予接管资格。变更部署或回退前先完成清理并归档；回退应用镜像时保留证据 PVC，重新核对绑定及校准适用性。

## K01–K07 现场验收与归档

先部署和确认 GPU、身份、独占，再验证局部性及设备时序，随后跑 A/A 和三分支。设备时序单项阻塞时，合格的动作回放仍可独立计分。

| 用例 | 操作与判据 | 归档 |
|---|---|---|
| K01 | 实际部署并经网关分别定向流式探测 A/B。逐 Service 去重后各有一个就绪后端。各容器可见一张授权完整物理 GPU，UUID 不同，核对 MIG、time-slicing 和 MPS。 | 清单与 digest、探测请求、EndpointSlice、Pod UID、节点、容器 ID、worker epoch、容器内 CUDA 设备和设备插件配置。 |
| K02 | A 预热后验证同卡第二 CUDA 上下文。以匹配工作量执行健康、A 争用、撤除三段，分别判断 A 慢化与恢复及 B、网关容差。 | 实际 UUID、计算模式、显存、进程和退出、计划与实际注入起止、三个阶段的原始样本、工作量桶及冻结容差。 |
| K03 | 在专用回合演练 readiness 摘除、worker 重启和网关账本中断，核对受影响运行进入配对排除，旧代次消息仅作审计。 | 变化前后绑定、端点与代次事件、证据覆盖缺口、原始网关游标和被排除轮次。 |
| K04 | 两个 runner 并发竞争同一 Lease，向持有者重发本轮启动并尝试跨尝试重置。只允许一个持有者和一份干扰进程。 | 两份独立尝试路径、Pod UID、持有者、版本与冲突记录、进程身份、幂等结果和跨尝试拒绝。 |
| K05 | 强制终止 runner，观察 A 侧硬停。旧持有者仍在时下轮阻塞。负责人确认清理、条件解除后，以新身份核对重置。 | runner 退出和硬停止时间、进程组退出、遗留请求及预留查证、保锁、负责人证据与版本、新尝试的复位结果。 |
| K06 | 演练正常、空闲、多迭代、标记未记录与错代、设备已完成而通知迟报、采样未完成而到达已过期。时钟误差跨时效或领先边界时标未知。另做采集开关对照。 | vLLM 构建和适配器、标记与覆盖、查询结果、采样/发出/网关接收/派发时间、时钟误差、延迟抖动、CPU/GPU 和遥测开销。设备核验和时效分别保存。 |
| K07 | 按冻结随机顺序独立 reset、预热、复跑 A/A 与 reference、immediate、delayed。验证同群与平台连续性，按计划档精确关联影子。 | 冻结研究与计划、到达/输入/预热/前史、独立启动及锚点、注入与绑定、原始结局、全池净增和 B 既有群新增违约。仅引擎进展时设备领先列留空并记录原因。 |

按 `runner_attempt_id/run_id/attempt_id` 隔离归档。保留 `preflight.json`、`platform_evidence.jsonl`、`manifest.json` 及其引用的原始文件、失败尝试和日志。时刻用本轮单调原点的整数 ns。注入标签与锁记录随离线证据保存。用原始文件字节重算 SHA-256，路径按 manifest 所在目录解析。保留 `metrics.csv`、`windows.csv`、`timeline.jsonl` 和两份状态文件。校准、探索和正式验证使用各自的独立回合和冻结配置。
