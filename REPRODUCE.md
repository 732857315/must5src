# 复现训练与浏览器构建

此仓库提供训练源码、冻结训练输入、训练/验证划分、初始化权重和发布权重。`must5` 仓库只放可部署的静态网站；这里的 `must5src` 用于检查、重训和重新导出。

2026-09-06 实测：在独立源码目录完整执行15轮训练，仍选中第12轮，9232条样本及训练/验证划分保持一致；46个参数张量在零容差下全部与发布模型相等。独立导出的34项运行资源也与线上构建逐字节一致，并通过25项数值对照。记录见 [reproduction/VERIFIED.json](reproduction/VERIFIED.json)。这些结果属于所列环境。

## 1. 安装当前验证环境

完整原生搜索与测试目前验证于 **Windows x64**。Python 原生搜索桥接使用 Windows DLL 和 `lld-link`；这不限制生成的网站在手机、电脑浏览器中运行。

当前复现环境为 Python **3.14.5**、Node.js **22.22.0**、npm **11.15.0**、LLVM **22.1.8**。Python 直接依赖见 [requirements-repro.txt](requirements-repro.txt)，浏览器依赖由 `package-lock.json` 固定为 ONNX Runtime Web 1.29.0。原实验未保存完整环境锁，不能把这份当前版本清单当成历史环境记录。

安装 Python、Node.js 和包含 `clang`、`wasm-ld`、`lld-link` 的 LLVM 后，在 PowerShell 执行。以下 LLVM 路径按默认安装位置填写，实际位置不同应替换：

```powershell
git clone https://github.com/732857315/must5src.git
cd must5src
py -3.14 -m venv .venv
$env:Path = "$PWD\.venv\Scripts;C:\Program Files\LLVM\bin;$env:Path"
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu
python -m pip install -r requirements-repro.txt
npm ci
python --version
node --version
clang --version
wasm-ld --version
lld-link --version
```

不需要 CUDA。旧 ncnn/pnnx 演示的可选依赖不属于上述 U-Net、全盘训练和浏览器导出流程。

## 2. 校验输入并重训当前全盘模型

`reproduction/manifest.json` 记录原始文件 SHA256 和训练代码身份。准备步骤校验输入，将证据引用与缓存报告的模型路径键重定位到当前克隆位置，原始输入保持原字节。默认生成 `reproduction/prepared_v2/`：缓存数据和划分只复制一次且 SHA 不变，模型路径变化前后按角色与 SHA 严格核对；训练入口使用该目录的新缓存报告。旧 `prepared/` 和失败输出保留，不能复用失败输出目录。

```powershell
python reproduction/prepare.py
python reproduction/run.py --output-dir training_runs/reproduced_v4 --dry-run
python reproduction/run.py --output-dir training_runs/reproduced_v4
```

输出目录必须不存在。默认运行原配置的 **15 轮**训练：从冻结的 global_v3 权重初始化，使用固定缓存、固定划分及经独立验证的行动约束；本地两个 U-Net 权重保持冻结。精确训练命令由 `--dry-run` 打印，训练配置、过程与选择结果保存在新输出目录。

只检查训练入口时可运行一轮，但这不是完整模型复现，学习率计划也与 15 轮不同：

```powershell
python reproduction/run.py --output-dir training_runs/smoke_v4 --epochs 1
```

比较重训权重与发布权重：

```powershell
python tools/browser/compare_checkpoints.py reproduction/inputs/global_policy_v4/global.pt training_runs/reproduced_v4/global.pt --atol 0 --rtol 0 --report training_runs/reproduced_v4/comparison_exact.json
```

此命令检查架构、参数名、形状、dtype 和每个张量，忽略运行路径、耗时等元信息。退出码 0 表示指定容差下通过，1 表示不匹配，2 表示输入错误。不同环境出现数值差异时可以另行记录明确容差，例如 `--atol 1e-6 --rtol 1e-5`，保留严格比较的失败报告，不能将容差结果写成逐位一致。

## 3. 导出与本地运行

下面从随仓库提供的发布权重构建；若要构建自己的重训结果，只将 `--global-checkpoint` 替换为新生成的 `global.pt`：

```powershell
python tools/browser/build.py --local-models reproduction/inputs/unet_curriculum_v2 --global-checkpoint reproduction/inputs/global_policy_v4/global.pt
python -m unittest discover -s tests -t . -p "test_*.py"
npm run test:browser
python tools/browser/serve.py --open
```

构建包含 25 项 PyTorch/ONNX 数值校验，并重新编译搜索 WASM。页面位于 `http://127.0.0.1:8770/`；可部署压缩包为 `exports/browser/must5-browser.zip`。浏览器测试应在构建之后运行，因为它们会读取生成的 WASM 和 ORT 文件。

安装在标准 Windows 路径的 Microsoft Edge 可用于实际浏览器检查。它会建立独立测试页面、验证推理、手机尺寸触控模拟、断网重载与项目子路径：

```powershell
python -m tests.browser.offline_probe --base-path must5
```

这里的手机尺寸模拟不代表 Android/iPhone 实机验证。对手机发布使用 GitHub Pages 等 HTTPS 静态站点；每台设备完成离线缓存后，模型和搜索均在该设备浏览器内计算。`127.0.0.1` 地址只供运行服务的本机访问。

## 4. 从头生成并训练两个 U-Net

`train_unet.py` 包含无禁下、单边禁下、任意禁下三阶段；在线扩增使用旋转、镜像以及颜色和行动方同时互换。随仓库保留各阶段基础样本、原配置、训练摘要与最终权重，位于 `reproduction/inputs/unet_curriculum_v2/`。

下面以原配置从新生成的数据训练两个模型，使用新的输出目录：

```powershell
python train_unet.py --output-dir training_runs/reproduced_unet --samples-per-stage 4096 --epochs 60 --batch-size 128 --base-channels 16 --threads 4 --learning-rate 0.002 --teacher-temperature 0.025 --patience 16 --seed 20260906
```

这个入口重新生成数据；当前全盘 v4 的可控重训使用随仓库固定的 U-Net 权重，不自动替换为新训练的 U-Net。若更换本地模型，应重新生成对应全盘特征数据，不能让不同权重与旧缓存混用。

## 复现范围

冻结输入、固定种子和训练代码支持检查同一训练流程。全盘老师使用受时间限制的搜索，硬件与负载会影响从零重新采集的对局和标签，因此固定缓存是当前 v4 重训的一部分。PyTorch 并未启用跨环境严格确定性保证；数值差异应通过比较报告公开。

数据 SHA 用于原始输入身份，权重张量比较用于训练结果，ONNX 校验用于导出结果。训练和数值校验均不构成必胜或比赛胜率证明。
