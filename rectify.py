"""
单/双目畸变矫正 (rectify)

单图模式 (默认, 兼容旧用法):
    python rectify.py -c config.yaml -l left01.jpg [-r right01.jpg] -v
    # 显示 left/right/concat 窗口, 写入 output/concated_rectify.jpg

Batch 模式 (按 pairRecord.txt 遍历):
    python rectify.py -c config.yaml --batch
    # 默认读 {output_dir}/pairRecord.txt, 找图按 config.root_dir 解析
    # 输出到 {output_dir}/rectified_samples/

    自定义路径:
    python rectify.py -c config.yaml --batch --pair_record path.txt --out_dir path/
"""
import argparse
import logging
import os
from pathlib import Path

import cv2

from src import parse_yaml, Camera
from src.utiles import read_pair_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_image(name, config, side):
    """根据 config.root_dir 找原始图. 双目走 left/right 子目录.
    name 可以带不带扩展名; 都接受."""
    root = Path(config.root_dir)
    if config.camera_num_type == "Stereo":
        sub = "left" if side == "L" else "right"
        d = root / sub
    else:
        d = root
    candidates = [name] if Path(name).suffix else [
        f"{name}{ext}" for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG")
    ]
    for c in candidates:
        p = d / c
        if p.exists():
            return p
    return None


def do_single(config, cam, left_path, right_path, verbose):
    left_img = cv2.imread(str(left_path))
    right_img = None
    if config.camera_num_type == "Stereo" and right_path is not None:
        right_img = cv2.imread(str(right_path))

    error = cam.compute_reprojection_errors()
    left, right, concat = cam.rectify(left_img, right_img)

    if verbose:
        cv2.imshow("left", left)
        if config.camera_num_type == "Stereo":
            cv2.imshow("right", right)
            cv2.imshow("concat", concat)
            cv2.imwrite(str(Path(config.output_dir) / "concated_rectify.jpg"), concat)
        cv2.waitKey()


def do_batch(config, cam, pair_record_path, out_dir):
    """按 pairRecord.txt 遍历, 矫正每对 (或每张) 并保存."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = read_pair_record(pair_record_path)
    if not record:
        logger.error(f"pairRecord 为空或不存在: {pair_record_path}")
        return
    is_stereo = isinstance(record[0], tuple)
    logger.info(f"Batch 模式: {'双目' if is_stereo else '单目'}, "
                f"共 {len(record)} {'对' if is_stereo else '张'}")

    for i, item in enumerate(record):
        if is_stereo:
            name_l, name_r = item
            p_l = _resolve_image(name_l, config, "L")
            p_r = _resolve_image(name_r, config, "R")
            if p_l is None or p_r is None:
                logger.warning(f"[{i+1}/{len(record)}] 跳过 (找不到图): {name_l} / {name_r}")
                continue
            left_img = cv2.imread(str(p_l))
            right_img = cv2.imread(str(p_r))
            left, right, concat = cam.rectify(left_img, right_img)
            cv2.imwrite(str(out_dir / f"{name_l}_rectified.png"), left)
            cv2.imwrite(str(out_dir / f"{name_r}_rectified.png"), right)
            if concat is not None:
                cv2.imwrite(str(out_dir / f"{name_l}_{name_r}_concat.jpg"), concat)
            logger.info(f"[{i+1}/{len(record)}] {name_l} + {name_r} -> {out_dir}")
        else:
            name = item
            p = _resolve_image(name, config, "L")
            if p is None:
                logger.warning(f"[{i+1}/{len(record)}] 跳过 (找不到图): {name}")
                continue
            img = cv2.imread(str(p))
            left, _, _ = cam.rectify(img, None)
            cv2.imwrite(str(out_dir / f"{name}_rectified.png"), left)
            logger.info(f"[{i+1}/{len(record)}] {name} -> {out_dir}")


def main(args):
    config = parse_yaml(args.config)
    cam = Camera(config)

    if args.batch:
        # 默认从 output_dir/pairRecord.txt 读
        if args.pair_record is None:
            args.pair_record = Path(config.output_dir) / "pairRecord.txt"
        # 默认输出到 output_dir/rectified_samples/
        if args.out_dir is None:
            args.out_dir = Path(config.output_dir) / "rectified_samples"
        do_batch(config, cam, args.pair_record, args.out_dir)
    else:
        do_single(config, cam, args.left_img, args.right_img, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单/双目畸变矫正")
    parser.add_argument("--config", "-c", default="config/configstereo.yaml")
    parser.add_argument("--verbose", "-v", default=False, action="store_true")
    parser.add_argument("--left_img", "-l",
                        default="E:/dataset/CameraCalib/public/bothImagesFixedStereo/left/left1.png")
    parser.add_argument("--right_img", "-r",
                        default="E:/dataset/CameraCalib/public/bothImagesFixedStereo/right/right1.png")
    parser.add_argument("--batch", "-b", default=False, action="store_true",
                        help="按 pairRecord.txt 批量矫正")
    parser.add_argument("--pair_record", default=None,
                        help="pairRecord.txt 路径, 默认 {output_dir}/pairRecord.txt")
    parser.add_argument("--out_dir", default=None,
                        help="batch 模式输出目录, 默认 {output_dir}/rectified_samples/")
    main(parser.parse_args())
