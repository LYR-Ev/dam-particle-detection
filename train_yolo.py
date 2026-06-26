r"""
YOLOv8 训练脚本 —— 堰塞坝表层颗粒检测

支持两阶段训练：
  quick  : yolov8n 快速验证（epochs=50, batch=8）
  final  : yolov8x 高精度最终训练（epochs=150, batch=4, patience=30）

特性：
  - 针对 RTX 3050 (4GB) 自动处理显存溢出（OOM 自动降 batch/imgsz 重试）
  - 训练完成自动把 best.pt 复制到 models/ 目录
  - 自动更新 config/config.yaml 中的模型路径
  - 训练完成后输出验证指标

使用方法（在自己的终端，用 .venv312 环境运行）：
  快速验证:  .\.venv312\Scripts\python.exe train_yolo.py --stage quick
  高精度:    .\.venv312\Scripts\python.exe train_yolo.py --stage final
  自定义:    .\.venv312\Scripts\python.exe train_yolo.py --model yolov8s.pt --epochs 100 --batch 4
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# 训练默认参数（对应你提供的命令）
# 默认使用项目内的数据集副本，沙箱环境下可正常写入 labels.cache
DATA_YAML = str(Path(__file__).parent / "dataset" / "data.yaml")
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
CONFIG_PATH = PROJECT_DIR / "config" / "config.yaml"

# 将 Ultralytics / matplotlib 的可写缓存目录重定向到项目内，
# 以便在沙箱（写入受限）环境下也能正常训练。
# 必须在 import ultralytics 之前设置，因为这些目录在 ultralytics 首次导入时就被初始化。
_SANDBOX_CACHE = PROJECT_DIR / ".cache"
(_SANDBOX_CACHE / "ultralytics").mkdir(parents=True, exist_ok=True)
(_SANDBOX_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(_SANDBOX_CACHE / "ultralytics"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_SANDBOX_CACHE / "ultralytics"))
os.environ.setdefault("MPLCACHEDIR", str(_SANDBOX_CACHE / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(_SANDBOX_CACHE / "torch"))
# 沙箱环境：禁止写入系统 Python 目录的 pycache（避免 worker 进程触发沙箱拦截）
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

STAGES = {
    "quick": {
        "model": "yolov8n.pt",
        "epochs": 50,
        "imgsz": 1024,
        "batch": 8,
        "patience": 20,
        "name": "rock_synthetic_v1",
    },
    "final": {
        "model": "yolov8x.pt",
        "epochs": 150,
        "imgsz": 1024,
        "batch": 4,
        "patience": 30,
        "name": "rock_synthetic_x_v1",
    },
}


def train_with_oom_retry(params, data_yaml):
    """
    执行训练，遇显存溢出(OOM)自动降低 batch / imgsz 重试。
    适用于 RTX 3050 4GB 显存较小的情况。
    """
    from ultralytics import YOLO

    model_path = params["model"]
    batch = params["batch"]
    imgsz = params["imgsz"]

    # 候选降级方案：逐步降低显存占用
    fallbacks = [
        {"batch": batch, "imgsz": imgsz},
        {"batch": max(1, batch // 2), "imgsz": imgsz},
        {"batch": max(1, batch // 2), "imgsz": imgsz // 2},
        {"batch": 1, "imgsz": imgsz // 2},
    ]

    for i, fb in enumerate(fallbacks):
        try:
            print(f"\n[训练尝试 {i+1}/{len(fallbacks)}] batch={fb['batch']}, imgsz={fb['imgsz']}")
            model = YOLO(model_path)
            results = model.train(
                data=data_yaml,
                epochs=params["epochs"],
                imgsz=fb["imgsz"],
                batch=fb["batch"],
                patience=params["patience"],
                augment=True,
                project=str(PROJECT_DIR / "runs" / "train"),
                name=params["name"],
                device=0,
                workers=0,
            )
            return results, fb
        except RuntimeError as e:
            err = str(e).lower()
            if "out of memory" in err or "cuda" in err and "memory" in err:
                print(f"[!] 显存不足 (OOM)，自动降低 batch/imgsz 重试...")
                torch_cuda_reset()
                continue
            else:
                raise
    raise RuntimeError("所有降级方案均失败，请检查显存或减小 imgsz 后重试。")


def torch_cuda_reset():
    """显存溢出后清理 CUDA 缓存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def update_config(best_pt_path):
    """更新 config/config.yaml 中的 yolo.model_path 指向新训练权重"""
    if not CONFIG_PATH.exists():
        print(f"[!] 配置文件不存在: {CONFIG_PATH}，跳过更新")
        return
    text = CONFIG_PATH.read_text(encoding="utf-8")
    # 替换 model_path 行（兼容带引号和不带引号两种写法）
    import re
    new_text = re.sub(
        r'(# YOLOv8 模型配置\nyolo:\n\s+model_path:\s*)("?)[^"\n]+("?)',
        rf'\1\2{best_pt_path}\3',
        text,
    )
    CONFIG_PATH.write_text(new_text, encoding="utf-8")
    print(f"[√] 已更新 config.yaml -> model_path: {best_pt_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 堰塞坝颗粒检测训练")
    parser.add_argument("--stage", choices=["quick", "final"], default="final",
                        help="训练阶段: quick(yolov8n快速) / final(yolov8x高精度), 默认 final")
    parser.add_argument("--model", default=None, help="自定义预训练模型 (覆盖 stage 默认值)")
    parser.add_argument("--epochs", type=int, default=None, help="自定义训练轮数")
    parser.add_argument("--batch", type=int, default=None, help="自定义 batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="自定义图像尺寸")
    parser.add_argument("--data", default=DATA_YAML, help="data.yaml 路径")
    args = parser.parse_args()

    data_yaml = args.data

    # 合并参数：stage 默认值 + 命令行覆盖
    params = STAGES[args.stage].copy()
    if args.model:
        params["model"] = args.model
    if args.epochs:
        params["epochs"] = args.epochs
    if args.batch:
        params["batch"] = args.batch
    if args.imgsz:
        params["imgsz"] = args.imgsz

    print("=" * 60)
    print("  YOLOv8 训练 —— 堰塞坝表层颗粒检测")
    print("=" * 60)
    print(f"  阶段:   {args.stage}")
    print(f"  模型:   {params['model']}")
    print(f"  轮数:   {params['epochs']}")
    print(f"  图像:   {params['imgsz']}px")
    print(f"  batch:  {params['batch']}")
    print(f"  数据:   {data_yaml}")
    print(f"  设备:   cuda:0 (GPU)")
    print("=" * 60)

    # 环境检查
    import torch
    print(f"\n[环境] torch={torch.__version__}, CUDA={torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f", GPU={torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"[环境] 显存: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("\n[!] 未检测到 CUDA，训练将无法使用 GPU。请先运行 setup_env.ps1")
        sys.exit(1)

    # 执行训练
    results, used = train_with_oom_retry(params, data_yaml)

    # 定位 best.pt
    best_pt = PROJECT_DIR / "runs" / "train" / params["name"] / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[!] 未找到 best.pt: {best_pt}")
        return

    print(f"\n[√] 训练完成，最佳权重: {best_pt}")

    # 复制到 models/
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / "best.pt"
    shutil.copy2(best_pt, dest)
    print(f"[√] 已复制到: {dest}")

    # 更新 config.yaml
    update_config("./models/best.pt")

    # 验证指标
    print("\n" + "=" * 60)
    print("  训练完成摘要")
    print("=" * 60)
    print(f"  实际使用 batch={used['batch']}, imgsz={used['imgsz']}")
    try:
        metrics_path = PROJECT_DIR / "runs" / "train" / params["name"] / "results.csv"
        if metrics_path.exists():
            import pandas as pd
            df = pd.read_csv(metrics_path)
            last = df.iloc[-1]
            print(f"  最终 mAP50:    {last.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  最终 mAP50-95: {last.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"  训练轮数:      {len(df)}")
    except Exception as e:
        print(f"  (指标读取失败: {e})")
    print("=" * 60)
    print("\n下一步：用新权重运行检测")
    print(f"  python main.py --image <你的影像> --scale 0.5")


if __name__ == "__main__":
    main()
