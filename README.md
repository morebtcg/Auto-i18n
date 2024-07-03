# Markdown自动翻译工具

## 快速上手

1. 将仓库克隆到本地。
2. 安装必需的模块：`pip install -r requirements.txt` 。
3. 执行命令 `python auto-translater.py` 运行程序，它将会递归处理测试目录 `testdir/to-translate` 下的所有 Markdown 文件，批量翻译为英语。

## 详细描述

程序 `auto-translater.py` 的运行逻辑如下：

1. 程序将递归处理测试目录 `testdir/to-translate` 下的所有 Markdown 文件，你可以在 `exclude_list` 变量中排除不需要翻译的文件。
2. 处理后的文件名会被记录在自动生成的 `processed_list.txt` 中。下次运行程序时，已处理的文件将不会再次翻译。