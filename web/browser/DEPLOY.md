# 五子棋 · 浏览器本地 AI

将发布包解压后的文件整体上传到同一个 HTTPS 静态站点目录。页面、模型和搜索都在访问者的浏览器内运行，无需推理服务器或外部 CDN。首次打开等待“离线已就绪”，之后可以断网刷新继续使用。开发时可用 localhost，直接打开 HTML 文件不支持完整运行。

默认棋盘为 15×15，行列可分别设置为 5–32；支持禁下编辑、悔棋、棋谱导出和手机放大棋盘。默认每步预算 1 秒，可设置为 0.1–30 秒，包含网络推理与搜索。超时未完成的验证会保持“未决”，不能据此判断落点安全。双方五子或更多相连获胜，双三、双四不是禁手。

棋局只保存在当前浏览器；清理网站数据会删除本机存档。保存失败时暂停对局，点击“重试保存”会继续提交原动作。离线缓存按站点目录隔离，不同部署可以共存。更新版本在旧页面关闭后接管，再次打开即可使用新资源。

默认只显示棋子，可在“落点参考”中开启预测叠加；“自定义禁下区域”和“查看分析详情”按需展开。方向键移动选点，Enter 或空格落子。新开对局会确认，取消后继续当前棋局；棋盘上方的下载按钮可导出棋谱。顶部 GitHub 入口和“开源与帮助”区提供源码、网页仓库及文档索引。

对局结束会弹出胜负或和棋提示，可选择“查看棋盘”保留终局，或“再来一局”立即开局并沿用执棋、尺寸、思考时间和禁下区域。刷新恢复已结束棋局时也会显示结果；关闭后不会因切换落点参考等操作重复弹出。

## 从源码构建

源码位于 [must5src](https://github.com/732857315/must5src)，环境安装见 [REPRODUCE.md](https://github.com/732857315/must5src/blob/main/REPRODUCE.md)。在已配置 Python、Node.js 和 LLVM 的源码目录执行：

```powershell
npm ci
npm run build:browser
python -m unittest discover -s tests -t . -p "test_*.py"
npm run test:browser
python -m tests.browser.offline_probe --base-path must5
python tools/browser/check_release.py
```

`npm run build:browser` 使用随源码提供的 `unet_curriculum_v2` 和 `global_policy_v4` 发布权重，执行 25 项 PyTorch/ONNX 数值校验并编译搜索 WASM。直接使用 `build.py` 时应显式指定与 npm 脚本相同的模型参数；其旧版默认路径仅用于兼容历史实验。

输出包为 `exports/browser/must5-browser.zip`，资源版本与 SHA-256 见 `assets.json`。发布包只包含声明的运行文件，不收录 `models/`、`vendor/` 中的历史残留。模型许可证和第三方声明保留在 `vendor/` 中。

## 发布与验证

```powershell
python tools/browser/prepare_pages.py
python tools/browser/prepare_pages.py --check --output exports/browser/pages-site
```

输出目录必须为空或尚不存在。将校验后的静态文件发布到 [must5](https://github.com/732857315/must5)，沿用该仓库的 GitHub Pages 设置。实际站点为 [risc.ink/must5](https://risc.ink/must5/)。重新打包时使用新的输出目录，避免混合版本。

部署后核对远端资源版本，并运行实际 HTTPS 地址的检查：

```powershell
python -m tests.browser.offline_probe --url https://risc.ink/must5/
```

两个局部模型和 16 个全局模型变体均须随包发布，支持全部方形和矩形尺寸。Worker 只保留最近使用的两个全局推理会话，加上两个局部会话，切换尺寸时释放不再使用的会话；磁盘离线缓存仍保留全部模型。棋盘在尺寸不变时复用格子和装饰节点。

浏览器功能、导出数值与手机尺寸触控模拟检查不等同于棋力或真实手机验收；当前模型尚未完成先后手各千局的棋力目标。训练、对局证据和数值对照工具保留在源码仓库中。
