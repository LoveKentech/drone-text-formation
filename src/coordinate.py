"""
텍스트 이미지를 래스터화하여 드론 목표 좌표(픽셀 → 2D 좌표)로 변환하는 모듈.
Pillow를 사용해 문자열을 렌더링하고, 활성 픽셀에서 대형 좌표를 추출한다.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import IMAGE_SIZE, SIZES


# size 문자열 → 배율 매핑
_SCALE = {"small": 1.0, "medium": 1.5, "large": 2.0}


def generate_coordinates(text: str, n: int, size: str) -> np.ndarray:
    """
    텍스트를 이미지로 렌더링한 뒤, 글자가 채워진 픽셀 위치 중
    n개를 균일하게 샘플링하여 드론 목표 좌표 집합 Q를 반환한다.

    Parameters
    ----------
    text : str
        렌더링할 문자열 (예: "AL", "LOVE", "KENTECH")
    n : int
        필요한 목표 좌표 수 (드론 수와 동일)
    size : str
        이미지 배율 — "small"(×1.0) / "medium"(×1.5) / "large"(×2.0)

    Returns
    -------
    np.ndarray, shape (n, 2)
        목표 좌표 배열. 각 행은 [x, y] (픽셀 단위).

    Raises
    ------
    ValueError
        size가 유효하지 않거나, 추출된 픽셀 수가 n보다 적을 때.

    Example
    -------
    # 드론 50대, "LOVE" 텍스트, 중간 크기로 목표 좌표 생성
    # Q = generate_coordinates("LOVE", 50, "medium")
    # print(Q.shape)  # (50, 2)
    """
    if size not in _SCALE:
        raise ValueError(f"size는 {list(_SCALE.keys())} 중 하나여야 합니다. 입력값: {size!r}")

    scale = _SCALE[size]
    w = int(IMAGE_SIZE[0] * scale)
    h = int(IMAGE_SIZE[1] * scale)

    # 흰 배경 이미지 생성 후 검은 글자 렌더링
    img = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)

    # 기본 폰트 크기는 이미지 높이의 75% 수준으로 설정
    font_size = max(10, int(h * 0.75))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        # 시스템에 Arial이 없을 경우 기본 비트맵 폰트 사용
        font = ImageFont.load_default()

    # 텍스트를 이미지 중앙에 배치
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0 = (w - text_w) // 2 - bbox[0]
    y0 = (h - text_h) // 2 - bbox[1]
    draw.text((x0, y0), text, fill=0, font=font)

    # 검은 픽셀(값 < 128) 좌표 추출 → shape (k, 2) [x, y]
    arr = np.array(img)
    ys, xs = np.where(arr < 128)
    pixels = np.column_stack([xs, ys])  # (k, 2)

    if len(pixels) < n:
        raise ValueError(
            f"추출된 픽셀 수({len(pixels)})가 요청한 드론 수({n})보다 적습니다. "
            f"size를 키우거나 n을 줄이세요."
        )

    # 균일 샘플링: 전체 픽셀에서 등간격으로 n개 선택
    indices = np.round(np.linspace(0, len(pixels) - 1, n)).astype(int)
    Q = pixels[indices].astype(float)

    return Q


# ---------------------------------------------------------------------------
# 단위 테스트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=== generate_coordinates 단위 테스트 ===\n")

    # 1) 기본 동작 확인
    for txt in ["AL", "LOVE", "KENTECH"]:
        for sz in SIZES:
            for n in [20, 50, 100]:
                Q = generate_coordinates(txt, n, sz)
                assert Q.shape == (n, 2), f"shape 오류: {Q.shape}"
                assert Q.dtype == float
    print("[PASS] 모든 (text, size, n) 조합에서 shape (n, 2) 반환")

    # 2) 좌표 범위 확인 — small 기준 이미지 크기 내에 있어야 함
    Q = generate_coordinates("AL", 20, "small")
    assert Q[:, 0].max() < IMAGE_SIZE[0], "x 좌표가 이미지 너비를 초과"
    assert Q[:, 1].max() < IMAGE_SIZE[1], "y 좌표가 이미지 높이를 초과"
    print("[PASS] 좌표 범위 이미지 크기 이내")

    # 3) 잘못된 size 입력 시 ValueError
    try:
        generate_coordinates("AL", 20, "huge")
        assert False, "ValueError가 발생해야 함"
    except ValueError:
        pass
    print("[PASS] 잘못된 size 입력 시 ValueError 발생")

    # 4) 시각화 확인 (matplotlib)
    Q = generate_coordinates("KENTECH", 100, "medium")
    plt.figure(figsize=(6, 3))
    plt.scatter(Q[:, 0], -Q[:, 1], s=10, c="steelblue")
    plt.title("generate_coordinates('KENTECH', 100, 'medium')")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("coordinate_test.png", dpi=100)
    plt.close()
    print("[INFO] coordinate_test.png 저장 완료")

    print("\n모든 테스트 통과.")
