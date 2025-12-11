#!/usr/bin/env python3
import os
import subprocess
import sys
import unittest


class EndToEndTest(unittest.TestCase):
    def test_full_pipeline_once(self):
        """执行一次完整流程并检查无错误退出"""
        # 测试音频文件
        audio_dir = os.path.join("tests", "test_audio")
        test_wav = os.path.join(audio_dir, "test.wav")
        self.assertTrue(os.path.exists(test_wav), f"测试音频不存在: {test_wav}")
        # 调用 orchestrator.py，使用配置文件并仅执行一次
        cmd = [
            sys.executable,
            os.path.join("auto_iteration", "orchestrator.py"),
            "--config",
            os.path.join(
                "auto_iteration", "orchestrator_config_example.json"
            ),
            "--override",
            "paths.raw_audio_dir=tests/test_audio",
            "iteration.once=true",
        ]
        result = subprocess.run(cmd)
        self.assertEqual(
            result.returncode, 0, f"管道执行失败，返回码: {result.returncode}"
        )


if __name__ == "__main__":
    unittest.main()
