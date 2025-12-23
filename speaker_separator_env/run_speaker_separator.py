#!/usr/bin/env python3

# 兼容入口：如果你已经把本目录安装为包（pip install -e .），推荐直接使用：
# - speaker-separator --audio ...
# - python -m speaker_separator_env --audio ...
#
# 这里保留 run_speaker_separator.py 作为“直接 python 运行”的入口。

from speaker_separator_env.cli import main


if __name__ == "__main__":
    main()

