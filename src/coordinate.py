"""
텍스트 이미지를 래스터화하여 드론 목표 좌표(픽셀 → 2D 좌표)로 변환하는 모듈.

세 가지 샘플링 방식을 지원한다:
  - "contour" : 글자 윤곽선을 따라 등간격으로 배치 (기본값)
  - "poisson"  : 글자 내부에 Poisson Disk 샘플링으로 배치
  - "grid"     : 글자 내부를 균등 격자로 샘플링 (구버전 방식)
"""

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import find_contours
from scipy.ndimage import distance_transform_cdt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import IMAGE_SIZE, CONTOUR_METHODS, DEFAULT_METHOD

# size 문자열 → 배율 매핑
_SCALE = {"small": 1.0, "medium": 1.5, "large": 2.0}

# 프로젝트 내 폰트 경로 (Noto Sans variable font — 모든 weight 포함)
_FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "fonts", "NotoSans-VF.ttf")

# Variable font에서 선택 가능한 weight (가는 순 → 굵은 순)
FONT_WEIGHTS = [
    "Thin", "ExtraLight", "Light", "Regular", "Medium",
    "SemiBold", "Bold", "ExtraBold", "Black",
]
DEFAULT_WEIGHT = "Black"  # 기존 동작 호환을 위한 기본값


# ===========================================================================
# 렌더링
# ===========================================================================

def _render_text_image(
    text: str,
    w: int,
    h: int,
    weight: str = DEFAULT_WEIGHT,
    font_path: str = _FONT_PATH,
) -> np.ndarray:
    """
    PIL로 text를 w×h grayscale 이미지로 렌더링한다.

    글자가 이미지 안에 꽉 차도록 폰트 크기를 자동으로 조정한다.
    Variable font의 named instance를 통해 weight를 선택한다.

    Parameters
    ----------
    text      : 렌더링할 문자열
    w         : 이미지 너비 (픽셀)
    h         : 이미지 높이 (픽셀)
    weight    : 폰트 굵기 — FONT_WEIGHTS 중 하나 (기본: "Black")
    font_path : 사용할 TTF 폰트 경로 (기본: fonts/NotoSans-VF.ttf)

    Returns
    -------
    np.ndarray, shape (h, w)
        글자=0(검정), 배경=255(흰색).

    Raises
    ------
    ValueError
        weight가 FONT_WEIGHTS에 없을 때.
    FileNotFoundError
        폰트 파일이 존재하지 않을 때.
    """
    if weight not in FONT_WEIGHTS:
        raise ValueError(
            f"weight는 {FONT_WEIGHTS} 중 하나여야 합니다. 입력값: {weight!r}"
        )

    abs_font = os.path.abspath(font_path)
    if not os.path.isfile(abs_font):
        raise FileNotFoundError(
            f"{abs_font} 파일이 없습니다. README를 참고하세요."
        )

    img = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)

    padding = 0.9
    chosen_size = 10
    for fs in range(max(10, int(h * 0.95)), 9, -2):
        font = ImageFont.truetype(abs_font, fs)
        font.set_variation_by_name(weight)
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= w * padding and (bbox[3] - bbox[1]) <= h * padding:
            chosen_size = fs
            break

    font = ImageFont.truetype(abs_font, chosen_size)
    font.set_variation_by_name(weight)
    bbox = draw.textbbox((0, 0), text, font=font)
    x0 = (w - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y0 = (h - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x0, y0), text, fill=0, font=font)

    return np.array(img)


# ===========================================================================
# 격자 샘플링 (Grid)  ―  구버전 방식
# ===========================================================================

def _make_grid_points(w: int, h: int, step: int) -> np.ndarray:
    """
    이미지 전체에 step 간격의 격자 좌표를 생성한다.

    격자는 (step//2, step//2)를 시작점으로 step 간격으로 배치된다.

    Parameters
    ----------
    w, h  : 이미지 크기 (픽셀)
    step  : 격자 간격 (픽셀)

    Returns
    -------
    np.ndarray, shape (k, 2)
        격자 좌표 배열 [x, y].
    """
    xs = np.arange(step // 2, w, step)
    ys = np.arange(step // 2, h, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _filter_grid_by_mask(grid_pts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    격자 좌표 중 마스크(글자 픽셀) 내부에 있는 것만 반환한다.

    Parameters
    ----------
    grid_pts : np.ndarray, shape (k, 2)
        격자 좌표 [x, y]
    mask     : np.ndarray, shape (h, w), dtype bool
        True = 글자 픽셀

    Returns
    -------
    np.ndarray, shape (m, 2)
        마스크 내부의 격자 좌표.
    """
    xs = grid_pts[:, 0].astype(int)
    ys = grid_pts[:, 1].astype(int)
    return grid_pts[mask[ys, xs]]


def _compute_grid_coverage(
    mask: np.ndarray, step: int, w: int, h: int
) -> float:
    """
    주어진 step의 격자가 마스크 픽셀을 얼마나 커버하는지 비율로 반환한다.

    Chebyshev 거리 변환을 사용해 각 마스크 픽셀에서 가장 가까운
    격자점까지의 거리를 계산하고, step//2 이내인 픽셀 비율을 반환한다.
    (binary_dilation 방식 대비 O(w×h)로 빠름)

    Parameters
    ----------
    mask       : 글자 픽셀 마스크
    step       : 격자 간격
    w, h       : 이미지 크기

    Returns
    -------
    float
        [0, 1] 범위의 커버리지 비율.
    """
    total = int(mask.sum())
    if total == 0:
        return 1.0

    grid_pts = _make_grid_points(w, h, step)
    if len(grid_pts) == 0:
        return 0.0

    pts_in_mask = _filter_grid_by_mask(grid_pts, mask)

    grid_bool = np.zeros((h, w), dtype=bool)
    valid_ys = pts_in_mask[:, 1].astype(int)
    valid_xs = pts_in_mask[:, 0].astype(int)

    grid_bool[valid_ys, valid_xs] = True

    if not np.any(grid_bool): return 0.0
    
    # 각 픽셀에서 가장 가까운 격자점까지 Chebyshev 거리
    dist = distance_transform_cdt(~grid_bool, metric="chessboard")
    covered = int((mask & (dist <= step // 2)).sum())
    return covered / total


def _find_grid_step(mask: np.ndarray, n: int, w: int, h: int) -> int:
    """
    마스크 내 격자점 수가 n개 이상이 되는 가장 큰 step을 이진 탐색으로 찾는다.

    step이 클수록 격자가 성기어 격자점 수가 줄어든다.
    가장 큰 step = 가장 균일한 배치로 n개를 만드는 최적 간격.

    Parameters
    ----------
    mask : 글자 픽셀 마스크
    n    : 필요한 최소 격자점 수
    w, h : 이미지 크기

    Returns
    -------
    int
        최적 격자 간격.
    """
    lo, hi = 1, max(w, h)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        pts = _filter_grid_by_mask(_make_grid_points(w, h, mid), mask)
        if len(pts) >= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _find_coverage_step(
    mask: np.ndarray, coverage: float, w: int, h: int
) -> int:
    """
    목표 커버리지를 달성하는 가장 큰 step을 이진 탐색으로 찾는다.

    Parameters
    ----------
    mask     : 글자 픽셀 마스크
    coverage : 목표 커버리지 비율 (0 < coverage <= 1.0)
    w, h     : 이미지 크기

    Returns
    -------
    int
        커버리지를 만족하는 가장 큰 격자 간격.
    """
    lo, hi = 1, max(w, h)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _compute_grid_coverage(mask, mid, w, h) >= coverage:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _sample_grid(mask: np.ndarray, n: int, w: int, h: int) -> np.ndarray:
    """
    글자 마스크 내부를 균등 격자로 샘플링하여 n개 좌표를 반환한다.

    이진 탐색으로 마스크 내 격자점이 n개 이상 나오는 가장 큰 step을 결정하고,
    그 격자점 중 등간격으로 n개를 선택한다.

    Parameters
    ----------
    mask : 글자 픽셀 마스크
    n    : 목표 좌표 수
    w, h : 이미지 크기

    Returns
    -------
    np.ndarray, shape (n, 2)

    Raises
    ------
    ValueError
        마스크 픽셀이 너무 적어 n개를 뽑을 수 없을 때.
    """
    step = _find_grid_step(mask, n, w, h)
    pts = _filter_grid_by_mask(_make_grid_points(w, h, step), mask)

    if len(pts) < n:
        raise ValueError(
            "글자가 너무 작거나 n이 너무 큽니다. size를 키우거나 n을 줄이세요."
        )

    indices = np.round(np.linspace(0, len(pts) - 1, n)).astype(int)
    return pts[indices].astype(float)


# ===========================================================================
# 윤곽선 샘플링 (Contour)
# ===========================================================================

def _extract_contour_points(arr: np.ndarray) -> np.ndarray:
    """
    grayscale 이미지 배열에서 글자 윤곽선 점들을 추출한다.

    skimage.measure.find_contours로 level=128 기준 윤곽선을 추출하고,
    여러 윤곽선(예: 'A'의 내부 구멍)을 모두 합쳐 하나의 배열로 반환한다.

    Parameters
    ----------
    arr : np.ndarray, shape (h, w)
        grayscale 이미지 배열 (글자=0, 배경=255)

    Returns
    -------
    np.ndarray, shape (k, 2)
        윤곽선 위의 점들 [x, y].
    """
    contours = find_contours(arr, level=128)
    if not contours:
        return np.empty((0, 2), dtype=float)

    # skimage는 (row, col) 순서 반환 → [x, y] = [col, row]로 변환
    parts = [np.column_stack([c[:, 1], c[:, 0]]) for c in contours]
    return np.vstack(parts)


def _sample_contour_equidistant(contour_pts: np.ndarray, n: int) -> np.ndarray:
    """
    윤곽선 점들에서 등간격으로 n개의 좌표를 샘플링한다.

    윤곽선 전체 길이를 연속 점들의 유클리드 거리 누적합으로 계산하고,
    np.interp로 목표 거리에 해당하는 x, y 좌표를 각각 보간한다.

    Parameters
    ----------
    contour_pts : np.ndarray, shape (k, 2)
        윤곽선 위의 점들 [x, y]
    n : int
        샘플링할 좌표 수

    Returns
    -------
    np.ndarray, shape (n, 2)

    Raises
    ------
    ValueError
        윤곽선 점 수가 n보다 적을 때.
    """
    if len(contour_pts) < n:
        raise ValueError(
            "글자가 너무 작거나 n이 너무 큽니다. size를 키우거나 n을 줄이세요."
        )

    seg_lengths = np.linalg.norm(np.diff(contour_pts, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    target_dists = np.linspace(0, cumlen[-1], n, endpoint=False)
    xs = np.interp(target_dists, cumlen, contour_pts[:, 0])
    ys = np.interp(target_dists, cumlen, contour_pts[:, 1])

    return np.column_stack([xs, ys])


# ===========================================================================
# Poisson Disk 샘플링
# ===========================================================================

def _sample_poisson_disk(mask: np.ndarray, n: int) -> np.ndarray:
    """
    Bridson 알고리즘 기반 Poisson Disk 샘플링으로 글자 내부를 채운다.

    r = sqrt(mask_pixel_count / (n * π * 0.5)) 로 최소 거리를 자동 계산하고,
    배경 격자(background grid)로 이웃 거리 검사를 O(1) 평균으로 최적화한다.

    Parameters
    ----------
    mask : np.ndarray, shape (h, w), dtype bool
        글자 내부 마스크 (True = 글자 픽셀)
    n    : 목표 샘플 수 (정확히 n개를 반환)

    Returns
    -------
    np.ndarray, shape (n, 2)

    Raises
    ------
    ValueError
        마스크 내 픽셀 수가 n보다 적을 때.
    """
    ys, xs = np.where(mask)
    pixel_count = len(xs)

    if pixel_count < n:
        raise ValueError(
            "글자가 너무 작거나 n이 너무 큽니다. size를 키우거나 n을 줄이세요."
        )

    h, w = mask.shape
    r = math.sqrt(pixel_count / (n * math.pi * 0.5))
    r = max(r, 1.0)

    cell = r / math.sqrt(2)
    grid_w = math.ceil(w / cell)
    grid_h = math.ceil(h / cell)
    grid: dict = {}  # (gi, gj) → [px, py]

    def _in_mask(px: float, py: float) -> bool:
        ix, iy = int(round(px)), int(round(py))
        return 0 <= ix < w and 0 <= iy < h and mask[iy, ix]

    def _is_far_enough(px: float, py: float) -> bool:
        gi = int(px / cell)
        gj = int(py / cell)
        for dj in range(max(0, gj - 2), min(grid_h, gj + 3)):
            for di in range(max(0, gi - 2), min(grid_w, gi + 3)):
                neighbor = grid.get((di, dj))
                if neighbor is not None:
                    dx = neighbor[0] - px
                    dy = neighbor[1] - py
                    if dx * dx + dy * dy < r * r:
                        return False
        return True

    rng = np.random.default_rng(seed=42)
    k_candidates = 30

    start_idx = int(rng.integers(0, pixel_count))
    p0 = [float(xs[start_idx]), float(ys[start_idx])]
    samples = [p0]
    active = [0]
    grid[(int(p0[0] / cell), int(p0[1] / cell))] = p0

    while active and len(samples) < n * 5:
        i = int(rng.integers(0, len(active)))
        ref = samples[active[i]]
        found = False

        for _ in range(k_candidates):
            angle = float(rng.uniform(0, 2 * math.pi))
            dist = float(rng.uniform(r, 2 * r))
            px = ref[0] + dist * math.cos(angle)
            py = ref[1] + dist * math.sin(angle)

            if _in_mask(px, py) and _is_far_enough(px, py):
                pt = [px, py]
                samples.append(pt)
                active.append(len(samples) - 1)
                grid[(int(px / cell), int(py / cell))] = pt
                found = True
                break

        if not found:
            active.pop(i)

    samples_arr = np.array(samples)

    # Poisson으로 부족할 경우 마스크 내 픽셀로 보충
    if len(samples_arr) < n:
        cands = np.column_stack([xs.astype(float), ys.astype(float)])
        tree = cKDTree(samples_arr)
        dists, _ = tree.query(cands, k=1)
        order = np.argsort(-dists)
        needed = n - len(samples_arr)
        samples_arr = np.vstack([samples_arr, cands[order[:needed]]])

    # 정확히 n개: 등간격 인덱스 선택
    if len(samples_arr) > n:
        indices = np.round(np.linspace(0, len(samples_arr) - 1, n)).astype(int)
        samples_arr = samples_arr[indices]

    return samples_arr.astype(float)


# ===========================================================================
# 공개 API
# ===========================================================================

def compute_optimal_n(
    text: str,
    size: str,
    coverage: float = 0.9,
    weight: str = DEFAULT_WEIGHT,
) -> int:
    """
    텍스트를 표현하는 데 필요한 최적 드론 수를 계산한다.

    격자 커버리지 기반 이진 탐색으로 결정한다:
    "텍스트 픽셀의 coverage 비율 이상을 Chebyshev 반경 step//2 내에서
    커버하는 가장 큰 step" → 그 step의 마스크 내 격자점 수를 반환.

    Parameters
    ----------
    text     : 렌더링할 문자열
    size     : 이미지 배율 — "small" / "medium" / "large"
    coverage : 목표 커버리지 비율 (기본값 0.9)
    weight   : 폰트 굵기 — FONT_WEIGHTS 중 하나 (기본: "Black")

    Returns
    -------
    int
        최적 드론 수 (≥ 1).

    Raises
    ------
    ValueError
        size 또는 weight가 유효하지 않을 때.
    FileNotFoundError
        폰트 파일이 없을 때.
    """
    if size not in _SCALE:
        raise ValueError(
            f"size는 {list(_SCALE.keys())} 중 하나여야 합니다. 입력값: {size!r}"
        )

    scale = _SCALE[size]
    w = int(IMAGE_SIZE[0] * scale)
    h = int(IMAGE_SIZE[1] * scale)

    arr = _render_text_image(text, w, h, weight=weight)
    mask = arr < 128

    if mask.sum() == 0:
        return 1

    step = _find_coverage_step(mask, coverage, w, h)
    pts = _filter_grid_by_mask(_make_grid_points(w, h, step), mask)
    return max(len(pts), 1)


def render_text_image(
    text: str, size: str, weight: str = DEFAULT_WEIGHT
) -> np.ndarray:
    """
    generate_coordinates와 동일한 조건으로 래스터화한 텍스트 이미지를 반환한다.

    visualize.py의 배경 이미지로 사용하면 드론 목표 좌표와 픽셀 단위로 정렬된다.

    Parameters
    ----------
    text   : 렌더링할 문자열
    size   : 이미지 배율 — "small" / "medium" / "large"
    weight : 폰트 굵기 — FONT_WEIGHTS 중 하나 (기본: "Black")

    Returns
    -------
    np.ndarray, shape (h, w)
        글자=0(검정), 배경=255(흰색).
    """
    if size not in _SCALE:
        raise ValueError(
            f"size는 {list(_SCALE.keys())} 중 하나여야 합니다. 입력값: {size!r}"
        )
    scale = _SCALE[size]
    w = int(IMAGE_SIZE[0] * scale)
    h = int(IMAGE_SIZE[1] * scale)
    return _render_text_image(text, w, h, weight=weight)


def generate_coordinates(
    text: str,
    n: int | None,
    size: str,
    method: str = DEFAULT_METHOD,
    weight: str = DEFAULT_WEIGHT,
) -> np.ndarray:
    """
    텍스트를 이미지로 렌더링한 뒤 지정된 방식으로 n개의 드론 목표 좌표를 반환한다.

    Parameters
    ----------
    text   : 렌더링할 문자열 (예: "AL", "LOVE", "KENTECH")
    n      : 목표 좌표 수. None이면 compute_optimal_n으로 자동 결정.
    size   : 이미지 배율 — "small"(×1.0) / "medium"(×1.5) / "large"(×2.0)
    method : 샘플링 방식 — "contour"(기본값) / "poisson" / "grid"
    weight : 폰트 굵기 — FONT_WEIGHTS 중 하나 (기본: "Black")

    Returns
    -------
    np.ndarray, shape (n, 2)
        목표 좌표 배열. 각 행은 [x, y] (픽셀 단위).

    Raises
    ------
    ValueError
        size / method / weight가 유효하지 않거나 점 수가 부족할 때.
    FileNotFoundError
        fonts/NotoSans-VF.ttf 파일이 없을 때.
    """
    if size not in _SCALE:
        raise ValueError(
            f"size는 {list(_SCALE.keys())} 중 하나여야 합니다. 입력값: {size!r}"
        )
    if method not in CONTOUR_METHODS:
        raise ValueError(
            f"method는 {CONTOUR_METHODS} 중 하나여야 합니다. 입력값: {method!r}"
        )

    if n is None:
        n = compute_optimal_n(text, size, weight=weight)

    scale = _SCALE[size]
    w = int(IMAGE_SIZE[0] * scale)
    h = int(IMAGE_SIZE[1] * scale)

    arr = _render_text_image(text, w, h, weight=weight)

    if method == "contour":
        contour_pts = _extract_contour_points(arr)
        Q = _sample_contour_equidistant(contour_pts, n)

    elif method == "poisson":
        mask = arr < 128
        Q = _sample_poisson_disk(mask, n)

    else:  # grid
        mask = arr < 128
        Q = _sample_grid(mask, n, w, h)

    return np.clip(Q, [0.0, 0.0], [w - 1.0, h - 1.0])


# ===========================================================================
# 단위 테스트
# ===========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion
    from config import DRONE_COUNTS, STRINGS, SIZES, CONTOUR_METHODS

    _OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "coordinate")
    os.makedirs(_OUT_DIR, exist_ok=True)

    print("=== generate_coordinates 단위 테스트 ===\n")

    # ── 1. 기본 동작: 모든 (text, size, n, method) 조합 ─────────────────────
    for txt in STRINGS:
        for sz in SIZES:
            for n in DRONE_COUNTS:
                for mth in CONTOUR_METHODS:
                    Q = generate_coordinates(txt, n, sz, mth)
                    assert Q.shape == (n, 2), (
                        f"shape 오류: {Q.shape} "
                        f"(txt={txt}, sz={sz}, n={n}, mth={mth})"
                    )
                    assert Q.dtype == float
    print("[PASS] 모든 (text, size, n, method) 조합에서 shape (n, 2) 반환")

    # ── 2. 좌표 범위 검증 ────────────────────────────────────────────────────
    for mth in CONTOUR_METHODS:
        Q = generate_coordinates(STRINGS[0], DRONE_COUNTS[0], "small", mth)
        assert Q[:, 0].max() < IMAGE_SIZE[0], f"[{mth}] x 좌표가 이미지 너비 초과"
        assert Q[:, 1].max() < IMAGE_SIZE[1], f"[{mth}] y 좌표가 이미지 높이 초과"
    print("[PASS] 좌표 범위 이미지 크기 이내")

    # ── 3. 윤곽선 방식 특성 검증(poisson, contour 방식)─────────────────────────────────────────────
    _txt, _sz, _n = "LOVE", "medium", DRONE_COUNTS[1]
    _w = int(IMAGE_SIZE[0] * 1.5)
    _h = int(IMAGE_SIZE[1] * 1.5)
    _arr = _render_text_image(_txt, _w, _h)
    _mask = _arr < 128

    Q_contour = generate_coordinates(_txt, _n, _sz, "contour")
    Q_poisson = generate_coordinates(_txt, _n, _sz, "poisson")

    # 3a) 윤곽선 점들이 글자 마스크 경계 근처에 있는지 확인
    # PIL anti-aliasing으로 level=128 윤곽선과 이진 마스크 경계가
    # 최대 ~10px 벌어질 수 있으므로 15px 여유를 허용한다.
    boundary = _mask & ~binary_erosion(_mask)
    _by, _bx = np.where(boundary)
    boundary_pts = np.column_stack([_bx.astype(float), _by.astype(float)])
    btree = cKDTree(boundary_pts)
    border_dists, _ = btree.query(Q_contour, k=1)
    assert border_dists.max() < 15.0, (
        f"윤곽선 좌표가 경계에서 너무 멀리 있습니다 "
        f"(max={border_dists.max():.2f}px)"
    )
    print("[PASS] 윤곽선 좌표가 글자 마스크 경계 근처에 위치")

    # 3b) 점 간 거리 표준편차: 윤곽선 < Poisson (등간격 → 균일도 높음)
    def _consecutive_dists(pts: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.diff(pts, axis=0), axis=1)

    std_contour = _consecutive_dists(Q_contour).std()
    std_poisson = _consecutive_dists(Q_poisson).std()
    assert std_contour < std_poisson, (
        f"윤곽선 방식의 균일도가 낮습니다 "
        f"(contour std={std_contour:.4f}, poisson std={std_poisson:.4f})"
    )
    print(
        f"[PASS] 점 간 거리 std: "
        f"contour={std_contour:.4f} < poisson={std_poisson:.4f}"
    )

    # ── 4. 잘못된 입력 예외 처리 ─────────────────────────────────────────────
    try:
        generate_coordinates(STRINGS[0], DRONE_COUNTS[0], "huge")
        assert False, "ValueError가 발생해야 함"
    except ValueError:
        pass
    print("[PASS] size='huge' → ValueError 발생")

    try:
        _render_text_image("AL", 200, 100, font_path="/nonexistent/font.ttf")
        assert False, "FileNotFoundError가 발생해야 함"
    except FileNotFoundError:
        pass
    print("[PASS] 폰트 파일 없을 때 → FileNotFoundError 발생")

    # ── 5. compute_optimal_n 검증 ────────────────────────────────────────────
    for txt in STRINGS:
        for sz in SIZES:
            n_opt = compute_optimal_n(txt, sz)
            assert isinstance(n_opt, int) and n_opt >= 1, (
                f"compute_optimal_n({txt!r}, {sz!r}) = {n_opt} (양의 정수여야 함)"
            )
    print("[PASS] compute_optimal_n 모든 조합에서 양의 정수 반환")

    n_al = compute_optimal_n("AL", "medium")
    n_kt = compute_optimal_n("KENTECH", "medium")
    assert n_kt > n_al, f"KENTECH({n_kt}) > AL({n_al}) 여야 함"
    print(f"[PASS] 단조성 — AL:{n_al}, KENTECH:{n_kt}")

    n_low = compute_optimal_n("LOVE", "medium", coverage=0.5)
    n_high = compute_optimal_n("LOVE", "medium", coverage=0.95)
    assert n_low <= n_high, (
        f"coverage 낮을수록 n이 작아야 함: low={n_low}, high={n_high}"
    )
    print(f"[PASS] coverage 파라미터 — 0.5→{n_low}드론, 0.95→{n_high}드론")

    # ── 6. n=None 자동 결정 ──────────────────────────────────────────────────
    for txt in STRINGS:
        for sz in SIZES:
            Q_auto = generate_coordinates(txt, None, sz)
            assert Q_auto.ndim == 2 and Q_auto.shape[1] == 2
            assert Q_auto.dtype == float
    print("[PASS] n=None 자동 결정 모드에서 shape (n, 2) 반환")

    # ── 7. 시각화: 세 방식 비교 PNG 저장 ────────────────────────────────────
    print("\n[INFO] 시각화 이미지 저장 중...")
    _SIZE_SCALE = {"small": 1.0, "medium": 1.5, "large": 2.0}
    _METHOD_LABEL = {
        "contour": "Contour",
        "poisson": "Poisson Disk",
        "grid": "Grid",
    }
    n_cols = len(DRONE_COUNTS)

    for txt in STRINGS:
        for sz in SIZES:
            w_v = int(IMAGE_SIZE[0] * _SIZE_SCALE[sz])
            h_v = int(IMAGE_SIZE[1] * _SIZE_SCALE[sz])
            arr_bg = _render_text_image(txt, w_v, h_v)

            for mth in CONTOUR_METHODS:
                label = _METHOD_LABEL[mth]
                fig, axes = plt.subplots(
                    1, n_cols + 1, figsize=(5 * (n_cols + 1), 4)
                )

                # 첫 열: 렌더링된 텍스트
                axes[0].imshow(arr_bg, cmap="gray", origin="upper")
                axes[0].set_title(f"Rendered Text\n({txt}, {sz})", fontsize=9)
                axes[0].axis("off")

                # 나머지 열: 드론 수별 오버레이
                for col, n in enumerate(DRONE_COUNTS, start=1):
                    Q = generate_coordinates(txt, n, sz, mth)
                    axes[col].imshow(
                        arr_bg, cmap="gray", origin="upper",
                        extent=[0, w_v, h_v, 0],
                    )
                    axes[col].scatter(
                        Q[:, 0], Q[:, 1],
                        s=12, c="steelblue", alpha=0.85, linewidths=0,
                    )
                    axes[col].set_xlim(0, w_v)
                    axes[col].set_ylim(h_v, 0)
                    axes[col].set_title(f"{label}\n{n} drones", fontsize=9)
                    axes[col].axis("off")

                plt.suptitle(f"{label} — {txt} / {sz}", fontsize=11)
                plt.tight_layout()

                fname = os.path.join(
                    _OUT_DIR, f"coordinate_{txt}_{sz}_{mth}.png"
                )
                plt.savefig(fname, dpi=120)
                plt.close()
                print(f"  [INFO] 저장: {fname}")

# ── 8. 시각화: compute_optimal_n 기반 최적 드론 배치 저장 ────────────────
    print("\n[INFO] 최적 드론 수(optimal n) 시각화 이미지 저장 중...")
    _SIZE_SCALE = {"small": 1.0, "medium": 1.5, "large": 2.0}
    # 별도 폴더 생성
    _OPT_DIR = os.path.join(_OUT_DIR, "optimal_n")
    os.makedirs(_OPT_DIR, exist_ok=True)

    for sz in SIZES:
        # 한 사이즈 내의 모든 스트링을 한눈에 보기 위해 서브플롯 구성
        fig, axes = plt.subplots(1, len(STRINGS), figsize=(5 * len(STRINGS), 5))
        if len(STRINGS) == 1: axes = [axes]

        for i, txt in enumerate(STRINGS):
            # 1. 최적 드론 수 계산 (coverage 기본값 0.9 적용)
            n_opt = compute_optimal_n(txt, sz)
            
            # 2. 좌표 생성 (기본 방식인 contour 사용)
            Q_opt = generate_coordinates(txt, n_opt, sz, method="grid")
            
            # 배경 렌더링
            w_v = int(IMAGE_SIZE[0] * _SIZE_SCALE[sz])
            h_v = int(IMAGE_SIZE[1] * _SIZE_SCALE[sz])
            arr_bg = _render_text_image(txt, w_v, h_v)

            # 시각화
            axes[i].imshow(arr_bg, cmap="gray", origin="upper", extent=[0, w_v, h_v, 0])
            axes[i].scatter(
                Q_opt[:, 0], Q_opt[:, 1], 
                s=15, c="orangered", edgecolors="white", linewidths=0.5, label="Drone"
            )
            axes[i].set_title(f"Text: {txt}\nOptimal n = {n_opt}", fontsize=12, fontweight='bold')
            axes[i].axis("off")

        plt.suptitle(f"Optimal Drone Formation (Size: {sz}, Coverage: 90%)", fontsize=16)
        plt.tight_layout()
        
        opt_fname = os.path.join(_OPT_DIR, f"optimal_n_{sz}.png")
        plt.savefig(opt_fname, dpi=130)
        plt.close()
        print(f"  [INFO] 저장 완료: {opt_fname}")

    print("\n모든 시각화 프로세스 종료.")

    print("\n모든 테스트 통과.")
