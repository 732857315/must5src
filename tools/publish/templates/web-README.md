# must5 · 浏览器五子棋

[打开网页](https://risc.ink/must5/) · [训练与构建源码](https://github.com/732857315/must5src)

手机、平板和电脑使用现代浏览器打开即可。AI 的模型推理和搜索都在当前设备内完成，无需连接个人电脑；首次联网等待“离线已就绪”后可断网继续下棋。浏览器清理网站数据后需要重新下载。

- 默认 15×15，行、列分别支持 5–32，兼容横纵矩形棋盘。
- 推算时间默认 1 秒，可设 0.1–30 秒。
- 黑白为棋子，灰色为禁下，红色为对手预测，绿色为推荐；热度各 25 级。
- 双方规则相同，五连或更长获胜；双三、双四属于需要防守的威胁。
- 两个 5×5 U-Net、一个全盘策略模型与 WebAssembly 搜索协同计算。

本仓库只保存网页运行所需资源。训练代码、固定数据、初始权重、复现命令见 must5src。模型仍处于实验阶段，没有通过“先手千局无负”等棋力验收。已通过 Windows Edge 桌面、手机触控模拟和断网检查；Android/iPhone 实机尚待验证。

## 参考与许可

项目开发参考了 [732857315/Gomoku-AI](https://github.com/732857315/Gomoku-AI)，其上游为 [whyb/Gomoku-AI](https://github.com/whyb/Gomoku-AI)。参考版本固定为 `bdfe39fa5aee404483976bfdfd03f13cfc6e585a`。早期开发使用参考仓库的 ONNX 作为对照模型；本站实际使用本项目训练的两个 U-Net 与全盘模型，不代表上游项目的模型或评测结果。

参考项目 BSD-3-Clause 许可保留在 [REFERENCE-LICENSE](REFERENCE-LICENSE)。ONNX Runtime Web 的许可和第三方声明保留在 [vendor/ONNX-RUNTIME-LICENSE](vendor/ONNX-RUNTIME-LICENSE) 和 [vendor/ONNX-RUNTIME-ThirdPartyNotices.txt](vendor/ONNX-RUNTIME-ThirdPartyNotices.txt)。
