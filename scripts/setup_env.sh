#!/usr/bin/env bash
# 在 Linux/macOS 或 WSL 中运行

# 1. 创建虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 升级 pip 并安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 提示
echo "虚拟环境已创建并安装依赖，使用 'source venv/bin/activate' 激活。"
