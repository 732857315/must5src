# 固定模型与 v4 训练复现材料

本目录用于在新检出目录中复现当前发布模型的导出，以及从**固定 v3 权重和固定数据**重新执行 v4 的 15 轮训练。它不重演历史全部对局，也不承诺跨系统重新训练后产生字节完全相同的 `.pt`/ONNX；比较模型应使用逐张量检查，不能把含有不同输出路径的 checkpoint 文件 SHA 当作唯一数值标准。训练成功不代表达到 2000 局棋力目标。

环境安装和完整流程见项目根目录的 [REPRODUCE.md](../REPRODUCE.md) 与 [requirements-repro.txt](../requirements-repro.txt)。这些依赖记录描述当前检查的复现环境；原训练未保存独立的依赖锁，不能追溯宣称它是历史环境锁。

## 先校验和准备，默认不训练

在项目根目录运行：

```bash
python reproduction/prepare.py
python reproduction/run.py --output-dir reproduction/runs/v4_full --dry-run
```

`prepare.py` 使用脚本所在项目根目录，逐项校验 `manifest.json` 中全部输入 SHA-256、大小以及训练器的 18 个本地 Python 依赖。它不访问原始 D 盘路径，不下载数据，不加载模型或搜索。三份约束 JSONL 的 `source_file`/`proof_file` 会生成相对路径副本，放到 `reproduction/prepared_v2/constraints/`。棋盘、行动方、标签、原棋谱/证书的 SHA 不变。棋谱和完整证书文件本身从未改写。

`reproduction/prepared_v2/data_source/` 保存一份原字节的 `dataset.jsonl.gz` 和 `split.json`。缓存报告仅将 `local_model_sha256` 的两个路径键改为当前检出目录中 `opponent.pt`、`play.pt` 的实际绝对路径；先按角色核验 SHA 与原记录逐一完全相等，其他报告内容不变，历史 `data_source.path` 仍保留作出处。原训练器会严格比较当前路径与 SHA，因此不能把原报告不加转换地直接传给新目录的训练。

`reproduction/prepared_v2/report.json` 记录新 `data_source`、原/新缓存报告 SHA、两个模型角色与原/新路径，以及数据和划分副本的相同 SHA。重复准备仅验证并复用相同文件，不重复复制 35 MB 缓存，拒绝替换不同的已有文件。旧 `reproduction/prepared/` 和既有失败结果保留；默认使用新的 `prepared_v2/`，两种准备目录均不进入源码发布。原先因绝对路径键不同而失败的训练不能续用原输出目录，应选择全新输出。准备过程只检查输入和引用，没有冒充完整防守树复验；实际训练器在训练前仍逐条独立验证全部防守证据。

## 在全新目录执行原 15 轮配置

```bash
python reproduction/run.py --output-dir reproduction/runs/v4_full
```

输出目录必须完全不存在，不能覆盖、续写旧结果。脚本自动完成校验和准备，将准备报告中的 `data_source` 传给根目录原 `train_global.py`，不改写冻结的 18 个训练源码文件。固定参数为：v3 权重初始化、相对 RGB、`proven_or_terminal` 标签、value loss 权重 0、15 epochs、学习率 0.0001、batch 32、4 线程、辅助图 dropout 0.3、seed 20260908、普通 searched CE 权重 0.1、约束训练曝光 16 倍、四项战术退化限制 0.02。优化器和学习率日程重新开始。

仅测试流程是否可跑时，可明确使用：

```bash
python reproduction/run.py --output-dir reproduction/runs/v4_smoke --epochs 1
```

此结果会标记为 `smoke_only_not_full_reproduction`，不能当作 15 轮复现或当前发布权重。除 `1` 与 `15` 外不接受其他轮次，避免把修改过的实验误报成固定配方。输出目录同级的 `<输出目录名>.invocation.json` 保留实际命令、manifest SHA、模式和完成状态；训练开始前目标目录保持为空，符合原训练器的初始化检查。

原实验：复用 v3 的 9,228 条缓存，保留 6,970/2,258 的训练/验证划分，再加四条已证败动作，最终为 6,973/2,259，训练曝光 7,018 条/epoch。原 15 轮保留第 12 轮。三份真实来源中有共同开局，留出源验证是有限方法实验，未知替代落点并未证明安全。原结果及初始化验证仅作为核对目标，不进入训练输入。

## 当前发布的 18 个 ONNX

两个 U-Net 和一个大局 checkpoint 总计约 1.10 MB，固定文件为：

- `inputs/unet_curriculum_v2/opponent.pt`
- `inputs/unet_curriculum_v2/play.pt`
- `inputs/global_policy_v4/global.pt`

安装环境、运行 `npm ci` 后，使用这三个发布权重导出：

```bash
python tools/browser/build.py --local-models reproduction/inputs/unet_curriculum_v2 --global-checkpoint reproduction/inputs/global_policy_v4/global.pt
```

构建入口支持 `--local-models` 后，上述命令无需把模型复制回被 Git 忽略的历史 `training_runs/` 路径。导出目标为 `web/browser/`，包括两个局部 ONNX 和 16 个逐轴池化变体，后者都来自同一个大局网络；变体不是 16 次训练。源码仓库用于训练和构建，`must5` 网页仓库只发布静态构建产物。

从本次新训练结果导出时，将 `--global-checkpoint` 改为新输出目录的 `global.pt`；应单独标记新模型身份，不覆盖“原发布权重”记录。逐张量检查可用：

```bash
python tools/browser/compare_checkpoints.py reproduction/inputs/global_policy_v4/global.pt reproduction/runs/v4_full/global.pt --atol 0 --rtol 0 --report reproduction/runs/v4_full/tensor_comparison.json
```

## 包含与不包含

`manifest.json` 是具体文件清单及 SHA 账本，不是需要用户填空的模板。目前 28 个输入文件共 42,474,463 字节，最大单件为 v3 压缩缓存 35,109,936 字节，均小于 100 MB。

| 内容 | 必要性 |
| --- | --- |
| v3 初始 PT、`dataset.jsonl.gz`、`dataset_report.json`、`split.json` | 固定 v4 初始状态、标签和划分；不依赖限时搜索重新生成同一轨迹 |
| 两个 U-Net PT | 保持缓存特征身份，以及为新约束生成相同局部特征 |
| 三个原始约束 JSONL、三个完整 `result.json`、四份完整证明 JSON | 独立核验实际棋谱来源和每条已证败动作 |
| v4 发布 PT、config/summary/initial_validation/split | 导出目标和原训练结果核对，不参与新训练 |
| U-Net 三阶段压缩 base、config/summary | 保留实际 12,288 个 base 样本与原训练配置，供研究和核对 |
| 原 plan/run JSON | 保留原来源 SHA、原始参数和完成记录；旧绝对路径只作历史元数据 |

不包含模型优化器历史、全部训练目录、比赛截图、浏览器 profile、逐步网络日志或对局视频。U-Net 数据生成器可按配置和 seed 新生成课程数据；当前 `train_unet.py` 没有从这三份 stage cache 直接训练的 CLI，因此这里不把“重新生成训练”伪称为已完成的缓存精确重训。v3 的历史原生搜索数据生成受时钟预算影响；seed 本身不能保证重生成同一数据，所以本配方提供实际缓存和 v3 初始权重。

## 来源与许可

原 v4 plan SHA-256：`c16618269466fc820c0bf1fe81cc68db835496717381a9e02e6460067d42a5de`。v3 数据 SHA：`d9ea90961e265715370f8ddc30211e831c22d83b884c1e23cc7f1d95149a34a2`。原 plan 没记录训练当时 Git commit，因此 manifest 的 `original_training_git_commit` 明确为 null，原训练身份由冻结源码 SHA 绑定；另记录整理时的源码提交 `99a82e76e7acc9b1eaaf08e740eabdc27039e3e4`，不冒充训练当时提交。

参考上游为 `https://github.com/732857315/Gomoku-AI.git`，固定提交 `bdfe39fa5aee404483976bfdfd03f13cfc6e585a`，BSD 3-Clause，`Copyright (c) 2024, whyb(張小凡)`。源码发布应保留原 BSD 许可正文和修改来源说明；不用把嵌套 `.repo/.git` 或上游所有运行产物放入复现材料。
