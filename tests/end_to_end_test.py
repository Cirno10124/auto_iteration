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
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        auto_iter_dir = os.path.dirname(tests_dir)
        audio_dir = os.path.join(tests_dir, "test_audio")
        test_wav = os.path.join(audio_dir, "test.wav")
        self.assertTrue(os.path.exists(test_wav), f"测试音频不存在: {test_wav}")
        # 调用 orchestrator.py，使用配置文件并仅执行一次
        cmd = [
            sys.executable,
            os.path.join(auto_iter_dir, "orchestrator.py"),
            "--config",
            os.path.join(auto_iter_dir, "orchestrator_config.json"),
            "--override",
            f"paths.raw_audio_dir={audio_dir}",
            f"paths.audio_dir={os.path.join(tests_dir, 'audio_chunks')}",
            f"paths.labels_dir={os.path.join(tests_dir, 'labels')}",
            f"paths.manifest_dir={os.path.join(tests_dir, 'manifests')}",
            f"paths.model_dir={os.path.join(tests_dir, 'model')}",
            f"paths.ggml_dir={os.path.join(tests_dir, 'ggml_model')}",
            f"logging.log_dir={os.path.join(tests_dir, 'logs')}",
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
            os.path.join(tests_dir, "audio_chunks"),
            os.path.join(tests_dir, "labels"),
            os.path.join(tests_dir, "manifests"),
            os.path.join(tests_dir, "model"),
            os.path.join(tests_dir, "ggml_model"),
            os.path.join(tests_dir, "logs"),
        ]:
            shutil.rmtree(d, ignore_errors=True)

    def test_two_iterations_use_previous_best_model(self):
        """执行两轮迭代，并确认第二轮标注/训练使用上一轮输出 best_model"""
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        auto_iter_dir = os.path.dirname(tests_dir)
        audio_dir = os.path.join(tests_dir, "test_audio")
        test_wav = os.path.join(audio_dir, "test.wav")
        self.assertTrue(os.path.exists(test_wav), f"测试音频不存在: {test_wav}")

        stubs_dir = os.path.join(tests_dir, "stubs")
        base_model = "DUMMY_BASE_MODEL"
        model_dir = os.path.join(tests_dir, "model")
        best_model_dir = os.path.join(model_dir, "best_model")

        cmd = [
            sys.executable,
            os.path.join(auto_iter_dir, "orchestrator.py"),
            "--config",
            os.path.join(auto_iter_dir, "orchestrator_config.json"),
            # 传入一个不存在的 speaker_id，使 speakers_map 过滤为空，从而走单说话人(None)路径，
            # 避免模型/日志输出被分散到 tests/model/<speaker_id>/... 导致断言路径不稳定。
            "--speakers",
            "__single_speaker__",
            "--override",
            f"paths.raw_audio_dir={audio_dir}",
            f"paths.split_script={os.path.join(stubs_dir, 'split_audio_stub.py')}",
            f"paths.labeler_script={os.path.join(stubs_dir, 'labeler_stub.py')}",
            f"paths.dataset_manager_script={os.path.join(stubs_dir, 'dataset_manager_stub.py')}",
            f"paths.train_script={os.path.join(stubs_dir, 'train_lora_stub.py')}",
            f"paths.evaluator_script={os.path.join(stubs_dir, 'evaluator_stub.py')}",
            f"paths.audio_dir={os.path.join(tests_dir, 'audio_chunks')}",
            f"paths.labels_dir={os.path.join(tests_dir, 'labels')}",
            f"paths.manifest_dir={os.path.join(tests_dir, 'manifests')}",
            f"paths.model_dir={model_dir}",
            f"paths.ggml_dir={os.path.join(tests_dir, 'ggml_model')}",
            f"logging.log_dir={os.path.join(tests_dir, 'logs')}",
            "iteration.once=false",
            "iteration.interval=0",
            "iteration.max_iterations=2",
            "iteration.skip_manifest=false",
            "iteration.stop_after_labels=false",
            f"labeling.model_name_or_path={base_model}",
            f"training.model_name_or_path={base_model}",
            "labeling.device=-1",
        ]
        result = subprocess.run(cmd)
        self.assertEqual(
            result.returncode, 0, f"管道执行失败，返回码: {result.returncode}"
        )

        # 断言：训练脚本收到两次调用；第二次使用 tests/model/best_model
        calls_path = os.path.join(model_dir, "training_calls.txt")
        self.assertTrue(os.path.exists(calls_path), "未生成训练调用记录")
        with open(calls_path, "r", encoding="utf-8") as f:
            calls = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        self.assertGreaterEqual(len(calls), 2, f"训练调用次数不足: {calls}")
        self.assertEqual(calls[0], base_model)
        self.assertEqual(calls[1], best_model_dir)

        # 断言：标注脚本第二轮同样使用上一轮 best_model
        labeler_calls_path = os.path.join(
            os.path.join(tests_dir, "labels"), "_labeler_calls.txt"
        )
        self.assertTrue(os.path.exists(labeler_calls_path), "未生成标注调用记录")
        with open(labeler_calls_path, "r", encoding="utf-8") as f:
            lcalls = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lcalls), 2, f"标注调用次数不足: {lcalls}")
        self.assertEqual(lcalls[0], base_model)
        self.assertEqual(lcalls[1], best_model_dir)

        for d in [
            os.path.join(tests_dir, "audio_chunks"),
            os.path.join(tests_dir, "labels"),
            os.path.join(tests_dir, "manifests"),
            os.path.join(tests_dir, "model"),
            os.path.join(tests_dir, "ggml_model"),
            os.path.join(tests_dir, "logs"),
        ]:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
