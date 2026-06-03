"""
评估功能测试脚本 — 验证 metrics.py 与各修改模块的正确性。

无需 GPU / 模型 / 标注数据，使用模拟数据即可运行全部测试。

运行方式:
    python test_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from metrics import (
    pixel_accuracy,
    compute_iou,
    dice_coefficient,
    compute_all_pixel_metrics,
    mean_iou,
    equivalent_circle_diameter,
    circularity,
    rosin_rammler_cdf,
    rosin_rammler_pdf,
    fit_rosin_rammler,
    compute_characteristic_sizes,
    calc_rr_characteristic_sizes,
    compute_mae,
    compute_rmse,
    compute_r_squared,
    compute_mre,
    compute_slope_intercept,
    generate_evaluation_report,
    print_evaluation_report,
    export_evaluation_csv,
    EvaluationReport,
)


def make_test_mask(size=100, circle=True, offset=(0, 0)):
    """生成模拟的圆形/方形测试掩码"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2 + offset[0], size // 2 + offset[1]
    radius = size // 4
    if circle:
        y, x = np.ogrid[:size, :size]
        mask[(x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2] = 1
    else:
        mask[
            cy - radius : cy + radius,
            cx - radius : cx + radius,
        ] = 1
    return mask


def test_pixel_accuracy():
    """测试像素精度计算"""
    print("--- 测试像素精度 (PA) ---")
    pred = make_test_mask(100)
    gt = make_test_mask(100)
    pa = pixel_accuracy(pred, gt)
    assert 0.9 <= pa <= 1.0, f"PA={pa} 应符合预期 (完全匹配)"
    print(f"  ✓ 完全匹配: PA = {pa:.4f}")

    gt_shifted = make_test_mask(100, offset=(5, 0))
    pa_partial = pixel_accuracy(pred, gt_shifted)
    assert 0.5 < pa_partial < 1.0, f"PA={pa_partial} 应反映部分匹配"
    print(f"  ✓ 部分匹配: PA = {pa_partial:.4f}")

    gt_empty = np.zeros((100, 100), dtype=np.uint8)
    pa_empty = pixel_accuracy(pred, gt_empty)
    assert 0.0 < pa_empty < 1.0
    print(f"  ✓ 空真值:   PA = {pa_empty:.4f}")
    print()


def test_iou():
    """测试 IoU 和 mIOU"""
    print("--- 测试 IoU / mIOU ---")
    pred = make_test_mask(100)
    gt = make_test_mask(100)
    iou = compute_iou(pred, gt)
    assert iou >= 0.99, f"完全匹配 IoU={iou}"
    print(f"  ✓ 完全匹配: IoU = {iou:.4f}")

    gt_shifted = make_test_mask(100, offset=(10, 0))
    iou_partial = compute_iou(pred, gt_shifted)
    assert 0.3 < iou_partial < 1.0
    print(f"  ✓ 部分匹配: IoU = {iou_partial:.4f}")

    gt_empty = np.zeros((100, 100), dtype=np.uint8)
    iou_empty = compute_iou(pred, gt_empty)
    assert iou_empty == 0.0, f"空真值 IoU={iou_empty} 应为 0"
    print(f"  ✓ 空真值:   IoU = {iou_empty:.4f}")

    miou = mean_iou([pred, pred], [gt, gt])
    assert miou >= 0.99
    print(f"  ✓ mIOU (2对): {miou:.4f}")
    print()


def test_dice():
    """测试 Dice 系数"""
    print("--- 测试 Dice 系数 ---")
    pred = make_test_mask(100)
    gt = make_test_mask(100)
    dice = dice_coefficient(pred, gt)
    assert dice >= 0.99
    print(f"  ✓ 完全匹配: Dice = {dice:.4f}")

    gt_empty = np.zeros((100, 100), dtype=np.uint8)
    dice_empty = dice_coefficient(pred, gt_empty)
    assert dice_empty == 0.0
    print(f"  ✓ 空真值:   Dice = {dice_empty:.4f}")
    print()


def test_all_pixel_metrics():
    """测试一次性计算所有像素级指标"""
    print("--- 测试 compute_all_pixel_metrics ---")
    pred_masks = [make_test_mask(100) for _ in range(5)]
    gt_masks = [make_test_mask(100) for _ in range(5)]
    metrics = compute_all_pixel_metrics(pred_masks, gt_masks)
    assert all(v >= 0.99 for v in metrics.values())
    for k, v in metrics.items():
        print(f"  ✓ {k}: {v:.4f}")
    print()


def test_equivalent_circle_diameter():
    """测试等效圆直径公式 — 论文公式(4)"""
    print("--- 测试等效圆直径 D = 2√(S/π) ---")
    S = np.pi * 25  # 面积 = πr², r=5 → D=10
    D = equivalent_circle_diameter(np.array([S]))
    assert abs(D[0] - 10.0) < 0.01, f"D={D[0]} 应约等于 10.0"
    print(f"  ✓ S={S:.2f}: D = {D[0]:.4f} (预期 10.0)")

    D_zero = equivalent_circle_diameter(np.array([0.0]))
    assert D_zero[0] == 0.0
    print(f"  ✓ S=0: D = {D_zero[0]:.4f} (预期 0.0)")
    print()


def test_circularity():
    """测试圆度计算"""
    print("--- 测试圆度 4πA/P² ---")
    r = 10.0
    A = np.pi * r ** 2
    P = 2 * np.pi * r
    circ = circularity(np.array([A]), np.array([P]))
    assert abs(circ[0] - 1.0) < 0.01, f"圆度={circ[0]} 应约等于 1.0"
    print(f"  ✓ 正圆: circularity = {circ[0]:.4f}")

    circ_nan = circularity(np.array([0.0]), np.array([0.0]))
    assert circ_nan[0] == 0.0
    print(f"  ✓ 零值: circularity = {circ_nan[0]:.4f}")
    print()


def test_rosin_rammler():
    """测试 Rosin-Rammler 分布拟合 — 论文公式(5)-(9)"""
    print("--- 测试 Rosin-Rammler 分布 ---")

    np.random.seed(42)
    Xm_true = 30.0
    n_true = 2.0
    n_samples = 200

    uniform_vals = np.random.uniform(0.001, 0.999, n_samples)
    diameters = Xm_true * ((-np.log(1.0 - uniform_vals) / 0.693) ** (1.0 / n_true))

    Xm, n, r2 = fit_rosin_rammler(diameters)
    assert 20 < Xm < 40, f"Xm={Xm} 应与 {Xm_true} 接近"
    assert 1.5 < n < 3.0, f"n={n} 应与 {n_true} 接近"
    assert r2 > 0.9, f"R²={r2} 应 > 0.9"
    print(f"  ✓ 拟合结果: Xm={Xm:.2f} (真值 {Xm_true}), n={n:.3f} (真值 {n_true}), R²={r2:.4f}")

    X_test = np.array([10, 20, 30, 40, 50])
    cdf_vals = rosin_rammler_cdf(X_test, Xm, n)
    assert np.all(cdf_vals >= 0) and np.all(cdf_vals <= 1)
    print(f"  ✓ CDF: X={X_test.tolist()} → R={np.round(cdf_vals, 4).tolist()}")

    pdf_vals = rosin_rammler_pdf(X_test, Xm, n)
    assert np.all(pdf_vals >= 0)
    print(f"  ✓ PDF: X={X_test.tolist()} → f={np.round(pdf_vals, 4).tolist()}")

    char_sizes = compute_characteristic_sizes(diameters)
    assert 10 in char_sizes and 50 in char_sizes and 100 in char_sizes
    print(f"  ✓ X10={char_sizes[10]:.1f}, X50={char_sizes[50]:.1f}, X90={char_sizes[90]:.1f}")

    rr_chars = calc_rr_characteristic_sizes(Xm, n)
    assert abs(rr_chars[50] - Xm) < 1.0
    print(f"  ✓ RR反推: X50={rr_chars[50]:.1f} (应≈Xm={Xm:.1f})")
    print()


def test_regression_metrics():
    """测试 MAE / RMSE / R² / MRE"""
    print("--- 测试回归评估指标 ---")
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    mae = compute_mae(pred, gt)
    assert mae == 0.0
    print(f"  ✓ 完全匹配: MAE = {mae:.4f}")

    rmse = compute_rmse(pred, gt)
    assert rmse == 0.0
    print(f"  ✓ 完全匹配: RMSE = {rmse:.4f}")

    r2 = compute_r_squared(pred, gt)
    assert r2 == 1.0
    print(f"  ✓ 完全匹配: R² = {r2:.4f}")

    mre = compute_mre(pred, gt)
    assert mre == 0.0
    print(f"  ✓ 完全匹配: MRE = {mre:.4f}")

    pred_bad = np.array([1.5, 2.5, 2.5, 3.5, 5.5])
    mae_bad = compute_mae(pred_bad, gt)
    rmse_bad = compute_rmse(pred_bad, gt)
    r2_bad = compute_r_squared(pred_bad, gt)
    assert mae_bad > 0
    assert rmse_bad > 0
    assert r2_bad < 1.0
    print(f"  ✓ 有偏差: MAE={mae_bad:.4f}, RMSE={rmse_bad:.4f}, R²={r2_bad:.4f}")

    slope, intercept, r_val = compute_slope_intercept(pred, gt)
    assert abs(slope - 1.0) < 0.01
    assert abs(intercept) < 0.01
    assert abs(r_val - 1.0) < 0.01
    print(f"  ✓ 回归: slope={slope:.4f}, intercept={intercept:.4f}, r={r_val:.4f}")
    print()


def test_evaluation_report():
    """测试完整评估报告生成（使用 dataclass 替代 ParticleMeasurement）"""
    print("--- 测试完整评估报告 ---")

    from metrics import EvaluationReport

    pred_masks = [make_test_mask(100) for _ in range(5)]
    gt_masks = [make_test_mask(100) for _ in range(5)]

    class MockMeasurement:
        def __init__(self, i):
            np.random.seed(42 + i)
            self.particle_id = i
            self.bbox = (0, 0, 100, 100)
            self.area_pixels = 490 + i * 10
            self.area_mm2 = 490.0 + i * 15.0
            self.long_axis_pixels = 25 + i * 2
            self.short_axis_pixels = 25 + i
            self.long_axis_mm = 25.0 + i * 3.0
            self.short_axis_mm = 25.0 + i * 1.5
            self.equivalent_diameter_pixels = 25.0 + i * 2.0
            self.equivalent_diameter_mm = 25.0 + i * 2.0
            self.axis_ratio = 1.0 + i * 0.1
            self.centroid = (50, 50)
            self.orientation_deg = i * 5.0
            self.perimeter_pixels = 78.5 + i * 3.0
            self.perimeter_mm = 78.5 + i * 3.0
            self.eq_circle_diameter_mm = 25.0 + i * 2.0
            self.circularity = max(0.1, 1.0 - i * 0.05)
            self.mask = make_test_mask(100)

    pred_meas = [MockMeasurement(i) for i in range(5)]
    gt_meas = [MockMeasurement(i) for i in range(5)]

    report = generate_evaluation_report(
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        pred_measurements=pred_meas,
        gt_measurements=gt_meas,
        image_name="test_image",
        scale_mm_per_pixel=1.0,
    )

    assert report.PA > 0.9, f"PA={report.PA}"
    print(f"  ✓ PA={report.PA:.4f}, mIOU={report.mIOU:.4f}, Dice={report.Dice:.4f}")

    report_text = print_evaluation_report(report)
    assert "Table 1" in report_text or "像素级" in report_text
    assert "Table 4" in report_text or "特征粒径" in report_text
    print("  ✓ 报告文本格式正确")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    csv_path = str(out_dir / "test_evaluation")
    export_evaluation_csv(report, csv_path)
    assert (out_dir / "test_evaluation_pixel_metrics.csv").exists()
    assert (out_dir / "test_evaluation_char_sizes.csv").exists()
    assert (out_dir / "test_evaluation_grain_level.csv").exists()
    print("  ✓ CSV 文件导出成功")

    # 清理测试文件
    for f in out_dir.glob("test_evaluation_*"):
        f.unlink()
    print()


def main():
    print("=" * 60)
    print("  评估功能测试套件 — Minerals 2024 论文指标体系")
    print("=" * 60)
    print()

    tests = [
        ("像素精度 (PA)", test_pixel_accuracy),
        ("IoU / mIOU", test_iou),
        ("Dice 系数", test_dice),
        ("所有像素级指标", test_all_pixel_metrics),
        ("等效圆直径 (Eq.4)", test_equivalent_circle_diameter),
        ("圆度 Circularity", test_circularity),
        ("Rosin-Rammler 分布 (Eq.5-9)", test_rosin_rammler),
        ("回归指标 (MAE/RMSE/R²/MRE)", test_regression_metrics),
        ("完整评估报告", test_evaluation_report),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name} 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name} 异常: {e}")
            failed += 1

    print("=" * 60)
    print(f"  测试结果: {passed} 通过, {failed} 失败 (共 {len(tests)} 项)")
    if failed == 0:
        print("  所有测试通过! 评估功能正常工作 ✓")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)