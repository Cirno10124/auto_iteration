#!/usr/bin/env python3
import os
import shutil
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
            os.path.join("orchestrator.py"),
            "--config",
            os.path.join("orchestrator_config_example.json"),
            "--override",
            "paths.raw_audio_dir=tests/test_audio",
            "paths.audio_dir=tests/audio_chunks",
            "paths.labels_dir=tests/labels",
            "paths.manifest_dir=tests/manifests",
            "paths.model_dir=tests/model",
            "paths.ggml_dir=tests/ggml_model",
            "logging.log_dir=tests/logs",
            "iteration.once=true",
            "iteration.skip_manifest=true",
            "iteration.stop_after_labels=true",
        ]
        result = subprocess.run(cmd)
        self.assertEqual(
            result.returncode, 0, f"管道执行失败，返回码: {result.returncode}"
        )
        # 清理测试生成的目录
        for d in [
            "tests/audio_chunks",
            "tests/labels",
            "tests/manifests",
            "tests/model",
            "tests/ggml_model",
            "tests/logs",
        ]:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
