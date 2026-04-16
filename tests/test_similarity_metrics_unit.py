"""纯数学：余弦相似度辅助函数，不依赖模型与数据，可在 CI 中稳定断言。"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(an, bn))


def test_cosine_same_direction_is_one():
    """同方向向量余弦相似度应接近 1。"""
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(_cosine(v, v * 3.0) - 1.0) < 1e-5


def test_cosine_opposite_direction_is_minus_one():
    """反方向向量余弦相似度应接近 -1。"""
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(_cosine(v, -v) - (-1.0)) < 1e-5


def test_cosine_orthogonal_is_near_zero():
    """正交向量余弦相似度应接近 0。"""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(_cosine(a, b)) < 1e-5
