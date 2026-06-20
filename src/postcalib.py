"""
标定后验证: 重投影误差统计 + 2D/3D 可视化
- 单目: 读 output/param.yaml, pose.txt, corners/, world_coordinates.csv
- 双目: 读 output/param.yaml, left_pose.txt/right_pose.txt, left_corners/right_corners/
"""
import os
import csv
import logging
from pathlib import Path
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")  # 非交互后端, 避免阻塞
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  触发 3d 投影注册
from matplotlib import cm
from matplotlib import font_manager as _fm

# 找系统里的 CJK 字体, 标题里的中文不变成方块
for _fp in (
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
):
    if os.path.exists(_fp):
        try:
            _fm.fontManager.addfont(_fp)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC",
                                    "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from .utiles import (
    parse_yaml,
    load_pose_file,
    load_corner_from_csv,
    getworldcornerpoints,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("calibration.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class PostCalibration:
    """读 output/ 产物, 出验证报告和图."""

    def __init__(self, config_path: str):
        self.config = parse_yaml(config_path)
        self.output_dir = Path(self.config.output_dir)
        self.verify_dir = self.output_dir / "verify"
        self.verify_dir.mkdir(parents=True, exist_ok=True)

        self.is_stereo = (self.config.camera_num_type == "Stereo")
        # 支持的传感器类型: Pinhole/Fisheye 用 cv2.projectPoints; Omnidir 暂不支持
        if self.config.camera_sensor_type not in ("Pinhole", "Fisheye"):
            raise NotImplementedError(
                f"暂不支持传感器类型 {self.config.camera_sensor_type}, "
                "目前仅支持 Pinhole / Fisheye"
            )

        # 读内参
        self._load_intrinsics()

        # 读世界点
        objp = getworldcornerpoints(self.config.board_size, self.config.square_size)
        self.world_points = objp.reshape(-1, 3).astype(np.float32)  # (N, 3)
        self.board_cols, self.board_rows = self.config.board_size  # (cols, rows)

        # 读位姿和角点 (双目: 字典里 key 加 'L_'/'R_' 前缀做区分)
        self.poses = {}     # {key: (rvec, tvec)}
        self.corners = {}   # {key: (N, 2) np.ndarray}
        self.images = {}    # {key: image_name}  用于 3D 标注
        self._load_poses_and_corners()

        # 算统计
        self.stats = self._compute_stats()

    # ------------------------------------------------------------------ load
    def _load_intrinsics(self):
        fs = cv2.FileStorage(str(self.output_dir / "param.yaml"), cv2.FILE_STORAGE_READ)
        try:
            self.sensor_type = fs.getNode("Camera_SensorType").string()
            self.width = int(fs.getNode("width").real())
            self.height = int(fs.getNode("height").real())
            if self.is_stereo:
                self.K = {"L": fs.getNode("K_l").mat(),
                          "R": fs.getNode("K_r").mat()}
                self.D = {"L": fs.getNode("D_l").mat(),
                          "R": fs.getNode("D_r").mat()}
                self.stereo_R = fs.getNode("R").mat()  # left -> right
                self.stereo_t = fs.getNode("t").mat()
            else:
                self.K = {"L": fs.getNode("K_l").mat()}
                self.D = {"L": fs.getNode("D_l").mat()}
        finally:
            fs.release()

    def _load_poses_and_corners(self):
        if self.is_stereo:
            pose_files = {"L": "left_pose.txt", "R": "right_pose.txt"}
            corner_dirs = {"L": "left_corners", "R": "right_corners"}
            for side, pf in pose_files.items():
                p = self.output_dir / pf
                if not p.exists():
                    logger.warning(f"未找到 {p}, 跳过 {side} 相机位姿")
                    continue
                poses = load_pose_file(p)
                for k, v in poses.items():
                    self.poses[f"{side}_{k}"] = v
                    self.images[f"{side}_{k}"] = f"{side}:{k}"
            for side, cd in corner_dirs.items():
                d = self.output_dir / cd
                if not d.exists():
                    logger.warning(f"未找到 {d}, 跳过 {side} 角点")
                    continue
                cs = load_corner_from_csv(str(d))
                for k, v in cs.items():
                    self.corners[f"{side}_{k}"] = v
        else:
            p = self.output_dir / "pose.txt"
            if not p.exists():
                raise FileNotFoundError(f"未找到 {p}")
            self.poses = load_pose_file(p)
            for k in self.poses:
                self.images[k] = k
            d = self.output_dir / "corners"
            if d.exists():
                self.corners = load_corner_from_csv(str(d))

    # -------------------------------------------------------------- compute
    def _compute_stats(self):
        per_image = []
        all_errs = []
        for key, (rvec, tvec) in self.poses.items():
            if key not in self.corners:
                logger.warning(f"无角点数据: {key}, 跳过统计")
                continue
            side = "L" if (key.startswith("L_") or not self.is_stereo) else "R"
            K = self.K[side]
            D = self.D[side]
            img_pts = self.corners[key].reshape(-1, 1, 2).astype(np.float32)
            obj_pts = self.world_points.reshape(-1, 1, 3).astype(np.float32)
            projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
            projected = projected.reshape(-1, 2)
            detected = img_pts.reshape(-1, 2)
            err = np.linalg.norm(projected - detected, axis=1)
            per_image.append({
                "key": key,
                "image": self.images[key],
                "side": side,
                "mean_px": float(err.mean()),
                "rmse_px": float(np.sqrt((err ** 2).mean())),
                "max_px": float(err.max()),
                "median_px": float(np.median(err)),
                "n_pts": int(len(err)),
            })
            all_errs.append(err)
        if all_errs:
            all_errs = np.concatenate(all_errs)
            overall = {
                "mean_px": float(all_errs.mean()),
                "rmse_px": float(np.sqrt((all_errs ** 2).mean())),
                "max_px": float(all_errs.max()),
                "median_px": float(np.median(all_errs)),
                "n_total": int(len(all_errs)),
            }
        else:
            all_errs = np.array([])
            overall = {"mean_px": 0, "rmse_px": 0, "max_px": 0, "median_px": 0, "n_total": 0}
        return {"per_image": per_image, "overall": overall, "all_errs": all_errs}

    # ------------------------------------------------------------------ csv
    def save_stats_csv(self):
        path = self.verify_dir / "reproj_stats.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "side", "mean_px", "rmse_px", "max_px", "median_px", "n_pts"])
            for r in self.stats["per_image"]:
                w.writerow([
                    r["image"], r["side"],
                    f"{r['mean_px']:.6f}", f"{r['rmse_px']:.6f}",
                    f"{r['max_px']:.6f}", f"{r['median_px']:.6f}", r["n_pts"],
                ])
            o = self.stats["overall"]
            w.writerow(["__overall__", "-",
                        f"{o['mean_px']:.6f}", f"{o['rmse_px']:.6f}",
                        f"{o['max_px']:.6f}", f"{o['median_px']:.6f}", o["n_total"]])
        logger.info(f"重投影统计已保存: {path}")

    # ------------------------------------------------------------------ 2D
    def plot_reprojection_per_image(self):
        rows = sorted(self.stats["per_image"], key=lambda r: r["rmse_px"])
        names = [r["image"] for r in rows]
        rmses = [r["rmse_px"] for r in rows]
        if not rows:
            logger.warning("无 per-image 数据, 跳过柱状图")
            return
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
        colors = cm.viridis(np.linspace(0, 1, len(rmses)))
        ax.bar(range(len(rmses)), rmses, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("RMSE (pixels)")
        ax.set_title(
            f"Per-image Reprojection RMSE  (overall = {self.stats['overall']['rmse_px']:.4f} px)"
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = self.verify_dir / "reproj_per_image.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"每图 RMSE 柱状图: {out}")

    def plot_reprojection_hist(self):
        all_errs = self.stats["all_errs"]
        if len(all_errs) == 0:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_errs, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(self.stats["overall"]["mean_px"], color="r", linestyle="--",
                   label=f"mean = {self.stats['overall']['mean_px']:.3f} px")
        ax.axvline(self.stats["overall"]["rmse_px"], color="g", linestyle="--",
                   label=f"rmse = {self.stats['overall']['rmse_px']:.3f} px")
        ax.set_xlabel("Reprojection Error (pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Reprojection Error Distribution (per point, all images)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out = self.verify_dir / "reproj_hist.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"误差直方图: {out}")

    def plot_reprojection_heatmap(self):
        # 平均每个角点位置的重投影误差, 画成 board (rows × cols) 热力图
        per_pt_err = np.zeros(self.board_cols * self.board_rows)
        per_pt_count = np.zeros(self.board_cols * self.board_rows)
        for key, (rvec, tvec) in self.poses.items():
            if key not in self.corners:
                continue
            side = "L" if (key.startswith("L_") or not self.is_stereo) else "R"
            K, D = self.K[side], self.D[side]
            img_pts = self.corners[key].reshape(-1, 1, 2).astype(np.float32)
            obj_pts = self.world_points.reshape(-1, 1, 3).astype(np.float32)
            projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
            err = np.linalg.norm(projected.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)
            per_pt_err += err
            per_pt_count += 1
        per_pt_err = per_pt_err / np.maximum(per_pt_count, 1)
        # 索引顺序: col 快变, row 慢变 -> reshape 成 (rows, cols) 网格
        err_grid = per_pt_err.reshape(self.board_rows, self.board_cols)
        fig, ax = plt.subplots(figsize=(max(6, self.board_cols), max(4, self.board_rows * 0.6)))
        im = ax.imshow(err_grid, cmap="hot", aspect="auto")
        ax.set_title("Mean Reprojection Error per Board Corner (pixel)")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        plt.colorbar(im, ax=ax, label="Error (px)")
        for i in range(self.board_rows):
            for j in range(self.board_cols):
                color = "cyan" if err_grid[i, j] > err_grid.mean() else "black"
                ax.text(j, i, f"{err_grid[i, j]:.2f}", ha="center", va="center",
                        color=color, fontsize=7)
        plt.tight_layout()
        out = self.verify_dir / "reproj_heatmap.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"角点误差热力图: {out}")

    # ------------------------------------------------------------------ 3D
    def _camera_center(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        return (-R.T @ tvec.flatten())

    def plot_poses_3d(self):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # 1) 棋盘格 (Z=0 平面)
        X = self.world_points[:, 0]
        Y = self.world_points[:, 1]
        Z = np.zeros_like(X)
        ax.scatter(X, Y, Z, c="green", s=25, alpha=0.7, label="Board corners")
        Xg = X.reshape(self.board_rows, self.board_cols)
        Yg = Y.reshape(self.board_rows, self.board_cols)
        Zg = Z.reshape(self.board_rows, self.board_cols)
        for i in range(self.board_rows):
            ax.plot(Xg[i], Yg[i], Zg[i], "g-", alpha=0.4, linewidth=0.6)
        for j in range(self.board_cols):
            ax.plot(Xg[:, j], Yg[:, j], Zg[:, j], "g-", alpha=0.4, linewidth=0.6)

        # 2) 世界坐标轴
        origin = np.zeros(3)
        L = self.config.square_size * 5
        ax.quiver(*origin, L, 0, 0, color="r", arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(*origin, 0, L, 0, color="g", arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(*origin, 0, 0, L, color="b", arrow_length_ratio=0.2, linewidth=2)
        ax.text(L * 1.1, 0, 0, "X", color="r")
        ax.text(0, L * 1.1, 0, "Y", color="g")
        ax.text(0, 0, L * 1.1, "Z", color="b")

        # 3) 相机位姿 (颜色按 RMSE)
        rmse_map = {r["key"]: r["rmse_px"] for r in self.stats["per_image"]}
        rmses = np.array(list(rmse_map.values())) if rmse_map else np.array([0])
        rmin, rmax = (rmses.min(), rmses.max()) if len(rmses) > 0 else (0, 1)
        ax_len = self.config.square_size * 3
        look_len = self.config.square_size * 10
        cam_centers = []
        for key, (rvec, tvec) in self.poses.items():
            R, _ = cv2.Rodrigues(rvec)
            c = -R.T @ tvec.flatten()
            cam_centers.append(c)
            side = "L" if (key.startswith("L_") or not self.is_stereo) else "R"
            marker = "o" if side == "L" else "^"
            rmse = rmse_map.get(key, 0)
            t = (rmse - rmin) / (rmax - rmin) if rmax > rmin else 0.5
            color = cm.viridis(t)
            ax.scatter(c[0], c[1], c[2], c=[color], s=80, marker=marker,
                       edgecolors="black", linewidths=0.5)
            # look-at 方向
            look = R.T @ np.array([0, 0, 1], dtype=np.float64)
            ax.quiver(c[0], c[1], c[2],
                      look[0] * look_len, look[1] * look_len, look[2] * look_len,
                      color=color, arrow_length_ratio=0.25, alpha=0.7, linewidth=1.2)
            # 相机三轴 (R,G,B = X,Y,Z 相机坐标)
            for vec, cc in zip(
                [R.T @ [1, 0, 0], R.T @ [0, 1, 0], R.T @ [0, 0, 1]],
                ["r", "g", "b"],
            ):
                ax.quiver(c[0], c[1], c[2],
                          vec[0] * ax_len, vec[1] * ax_len, vec[2] * ax_len,
                          color=cc, arrow_length_ratio=0.3, alpha=0.5, linewidth=0.8)

        # 4) 视角: 等比 + 居中
        if cam_centers:
            cam_arr = np.array(cam_centers)
            board_pts = np.column_stack([X, Y, Z])
            combined = np.vstack([cam_arr, board_pts])
            ranges = combined.max(axis=0) - combined.min(axis=0)
            center = (combined.max(axis=0) + combined.min(axis=0)) / 2
            max_range = max(ranges.max() / 2, 1e-3)
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Camera Poses & World Board (color = reprojection RMSE)")
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        out = self.verify_dir / "poses_3d.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"3D 位姿图: {out}")

    def plot_stereo_baseline(self):
        """双目: 取左右各自第一张图, 算 baseline + 俯视相对位姿."""
        if not self.is_stereo:
            return
        if not hasattr(self, "stereo_R"):
            logger.warning("param.yaml 中无 R/t, 跳过 baseline 图")
            return
        fig, ax = plt.subplots(figsize=(8, 8))
        R_lr = self.stereo_R
        t_lr = self.stereo_t.flatten()
        baseline = float(np.linalg.norm(t_lr))
        # 左相机
        ax.scatter(0, 0, c="red", s=160, marker="o", edgecolors="black", label="Left cam")
        ax.arrow(0, 0, 0.05, 0, head_width=0.02, head_length=0.01, fc="r", ec="r")
        ax.text(0.06, 0.0, "L", color="r", fontsize=14, va="center")
        # 右相机
        ax.scatter(t_lr[0], t_lr[1], c="blue", s=160, marker="^",
                   edgecolors="black", label="Right cam")
        # 朝向: R_lr 第一列 (左相机 X 轴在世界系下的方向) 作为朝向示意
        ax.arrow(t_lr[0], t_lr[1], R_lr[0, 0] * 0.05, R_lr[1, 0] * 0.05,
                 head_width=0.02, head_length=0.01, fc="b", ec="b")
        ax.text(t_lr[0] + 0.06, t_lr[1], "R", color="b", fontsize=14, va="center")
        # baseline 线
        ax.plot([0, t_lr[0]], [0, t_lr[1]], "k--", alpha=0.6)
        ax.text(t_lr[0] / 2, t_lr[1] / 2,
                f"baseline = {baseline * 1000:.2f} mm",
                fontsize=11, color="purple",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="purple"))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Stereo Baseline (left → right)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        out = self.verify_dir / "stereo_baseline.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"双目 baseline 图: {out}")

    # -------------------------------------------------------------- summary
    def write_summary(self):
        path = self.verify_dir / "summary.txt"
        o = self.stats["overall"]
        rows = self.stats["per_image"]
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== 重投影误差统计 ===\n")
            f.write(f"模式:           {self.config.camera_num_type}\n")
            f.write(f"传感器类型:     {self.config.camera_sensor_type}\n")
            f.write(f"图像数:         {len(rows)}\n")
            f.write(f"总角点数:       {o['n_total']}\n")
            f.write(f"平均误差:       {o['mean_px']:.6f} px\n")
            f.write(f"RMSE:           {o['rmse_px']:.6f} px\n")
            f.write(f"中位数:         {o['median_px']:.6f} px\n")
            f.write(f"最大单点误差:   {o['max_px']:.6f} px\n")
            if rows:
                worst = max(rows, key=lambda r: r["rmse_px"])
                best = min(rows, key=lambda r: r["rmse_px"])
                f.write(f"最差图:         {worst['image']}  RMSE = {worst['rmse_px']:.4f} px\n")
                f.write(f"最好图:         {best['image']}  RMSE = {best['rmse_px']:.4f} px\n")
            f.write("\n=== 相机位姿 3D 散布 ===\n")
            cam_pts = []
            for k, (rvec, tvec) in self.poses.items():
                cam_pts.append(self._camera_center(rvec, tvec))
            cam_pts = np.array(cam_pts) if cam_pts else np.zeros((0, 3))
            if len(cam_pts) > 0:
                f.write(f"X 范围: [{cam_pts[:, 0].min():.4f}, {cam_pts[:, 0].max():.4f}] m"
                        f"  跨度 {cam_pts[:, 0].ptp():.4f} m\n")
                f.write(f"Y 范围: [{cam_pts[:, 1].min():.4f}, {cam_pts[:, 1].max():.4f}] m"
                        f"  跨度 {cam_pts[:, 1].ptp():.4f} m\n")
                f.write(f"Z 范围: [{cam_pts[:, 2].min():.4f}, {cam_pts[:, 2].max():.4f}] m"
                        f"  跨度 {cam_pts[:, 2].ptp():.4f} m\n")
            if self.is_stereo and hasattr(self, "stereo_t"):
                f.write(f"\n双目 baseline:  {np.linalg.norm(self.stereo_t) * 1000:.2f} mm\n")
        logger.info(f"汇总: {path}")

    # ------------------------------------------------------------------ run
    def run(self):
        self.save_stats_csv()
        self.plot_reprojection_per_image()
        self.plot_reprojection_hist()
        self.plot_reprojection_heatmap()
        self.plot_poses_3d()
        if self.is_stereo:
            self.plot_stereo_baseline()
            self.plot_epipolar_before()
            self.plot_rectified_pair()
        self.write_summary()
        logger.info(f"验证完成, 产物目录: {self.verify_dir}")

    # ----------------------------------------------- stereo image helpers
    def _find_first_pair(self):
        """找同时有左右角点的第一对. 返回 (id, name_l, name_r), id 是后缀数字."""
        import re
        left_by_id, right_by_id = {}, {}
        for k in self.corners:
            m = re.search(r"(\d+)$", k)
            if not m:
                continue
            sid = m.group(1)
            if k.startswith("L_"):
                left_by_id[sid] = k[2:]
            elif k.startswith("R_"):
                right_by_id[sid] = k[2:]
        common = sorted(set(left_by_id) & set(right_by_id))
        if not common:
            return None
        sid = common[0]
        return sid, left_by_id[sid], right_by_id[sid]

    def _resolve_image_path(self, name, side):
        """根据 config.root_dir 找原始图 (双目的 left/right 子目录)."""
        if self.is_stereo:
            sub = "left" if side == "L" else "right"
            d = Path(self.config.root_dir) / sub
        else:
            d = Path(self.config.root_dir)
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"):
            p = d / f"{name}{ext}"
            if p.exists():
                return p
        return None

    def _fundamental_matrix(self):
        K_l, K_r = self.K["L"], self.K["R"]
        R, t = self.stereo_R, self.stereo_t.flatten()
        t_cross = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0],
        ])
        E = t_cross @ R
        F = np.linalg.inv(K_r).T @ E @ np.linalg.inv(K_l)
        return F

    # -------------------------------------------------------- epipolar pre
    def plot_epipolar_before(self):
        """校正前: 在 undistort 后左图画角点, 右图画对应极线 + 真实匹配点."""
        pair = self._find_first_pair()
        if pair is None:
            logger.warning("无有效左右图像对, 跳过 epipolar_before")
            return
        _, name_l, name_r = pair
        p_l = self._resolve_image_path(name_l, "L")
        p_r = self._resolve_image_path(name_r, "R")
        if p_l is None or p_r is None:
            logger.warning(f"找不到图像: {p_l} 或 {p_r}")
            return

        img_l = cv2.imread(str(p_l))
        img_r = cv2.imread(str(p_r))
        K_l, D_l = self.K["L"], self.D["L"]
        K_r, D_r = self.K["R"], self.D["R"]
        F = self._fundamental_matrix()

        # 只 undistort, 不 rectify
        img_l_und = cv2.undistort(img_l, K_l, D_l)
        img_r_und = cv2.undistort(img_r, K_r, D_r)
        h, w = img_r_und.shape[:2]

        pts_l = self.corners.get(f"L_{name_l}")
        pts_r = self.corners.get(f"R_{name_r}")
        if pts_l is None or pts_r is None:
            logger.warning("角点缺失, 跳过 epipolar_before")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].imshow(cv2.cvtColor(img_l_und, cv2.COLOR_BGR2RGB))
        axes[0].scatter(pts_l[:, 0], pts_l[:, 1], s=30, c="red",
                        edgecolors="yellow", linewidths=0.7, zorder=3)
        axes[0].set_title(f"Left (undistorted) — {name_l}\n"
                          f"detected corners: {len(pts_l)}")
        axes[0].set_xlim(0, w - 1)
        axes[0].set_ylim(h - 1, 0)
        axes[0].axis("off")

        # 右图: 画极线 (红线) + 真实匹配点 (绿圈)
        axes[1].imshow(cv2.cvtColor(img_r_und, cv2.COLOR_BGR2RGB))
        axes[1].scatter(pts_r[:, 0], pts_r[:, 1], s=30, c="lime",
                        edgecolors="black", linewidths=0.7, zorder=3,
                        label="matched right corner")
        # 极线: l[0]x + l[1]y + l[2] = 0  →  x = (-l[1]y - l[2]) / l[0]
        # 端点用 x 在 [0, w-1] 范围内采样, 避免 F 未归一化时坐标爆掉
        n_samples = 80
        xs = np.linspace(0, w - 1, n_samples)
        for p in pts_l:
            line = F @ np.array([p[0], p[1], 1.0])
            if abs(line[0]) < 1e-9:
                continue
            ys = (-line[0] * xs - line[2]) / line[1]
            # 只画落在 [0, h-1] 范围内的连续段
            mask = (ys >= 0) & (ys <= h - 1)
            if not mask.any():
                continue
            # 找 mask 连续段
            seg_start = None
            for i in range(n_samples):
                if mask[i] and seg_start is None:
                    seg_start = i
                elif not mask[i] and seg_start is not None:
                    axes[1].plot(xs[seg_start:i], ys[seg_start:i], "r-",
                                 alpha=0.4, linewidth=0.7, zorder=2)
                    seg_start = None
            if seg_start is not None:
                axes[1].plot(xs[seg_start:], ys[seg_start:], "r-",
                             alpha=0.4, linewidth=0.7, zorder=2)
        axes[1].set_xlim(0, w - 1)
        axes[1].set_ylim(h - 1, 0)
        axes[1].set_title(f"Right (undistorted) — epipolar lines from LEFT corners\n"
                          f"(red = l_r = F·p_l,  green = real matched)")
        axes[1].axis("off")
        axes[1].legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        out = self.verify_dir / "epipolar_before.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"校正前极线图: {out}")

    # ------------------------------------------------------- rectified
    def plot_rectified_pair(self):
        """校正后: stereoRectify + remap, 画水平线检查 y 坐标对齐."""
        pair = self._find_first_pair()
        if pair is None:
            return
        _, name_l, name_r = pair
        p_l = self._resolve_image_path(name_l, "L")
        p_r = self._resolve_image_path(name_r, "R")
        if p_l is None or p_r is None:
            return
        K_l, D_l = self.K["L"], self.D["L"]
        K_r, D_r = self.K["R"], self.D["R"]
        R, t = self.stereo_R, self.stereo_t
        w, h = self.width, self.height
        sensor = self.config.camera_sensor_type

        try:
            if sensor == "Pinhole":
                R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                    K_l, D_l, K_r, D_r, (w, h), R, t,
                    flags=0, alpha=self.config.alpha,
                )
                m1l, m2l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
                m1r, m2r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)
            elif sensor == "Fisheye":
                R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
                    K_l, D_l, K_r, D_r, (w, h), R, t, flags=0,
                )
                m1l, m2l = cv2.fisheye.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
                m1r, m2r = cv2.fisheye.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)
            else:
                logger.warning(f"sensor {sensor} 的 rectify 在 verify 中暂未实现")
                return
        except cv2.error as e:
            logger.warning(f"stereoRectify 失败: {e}")
            return

        img_l = cv2.imread(str(p_l))
        img_r = cv2.imread(str(p_r))
        rect_l = cv2.remap(img_l, m1l, m2l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, m1r, m2r, cv2.INTER_LINEAR)
        pts_l = self.corners.get(f"L_{name_l}")
        pts_r = self.corners.get(f"R_{name_r}")

        fig, axes = plt.subplots(2, 1, figsize=(14, 11))
        axes[0].imshow(cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB))
        if pts_l is not None:
            axes[0].scatter(pts_l[:, 0], pts_l[:, 1], s=20, c="red",
                            edgecolors="yellow", linewidths=0.5, zorder=3)
            # 水平线 (校正后极线应该是水平的)
            for y in pts_l[:, 1]:
                axes[0].axhline(y, color="cyan", linestyle="--",
                                alpha=0.5, linewidth=0.5, zorder=2)
        axes[0].set_title(f"Rectified Left — {name_l}\n"
                          f"(cyan dashed = ideal horizontal epipolar lines)")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(rect_r, cv2.COLOR_BGR2RGB))
        if pts_r is not None:
            axes[1].scatter(pts_r[:, 0], pts_r[:, 1], s=20, c="lime",
                            edgecolors="black", linewidths=0.5, zorder=3)
            for y in pts_r[:, 1]:
                axes[1].axhline(y, color="cyan", linestyle="--",
                                alpha=0.5, linewidth=0.5, zorder=2)
        axes[1].set_title(f"Rectified Right — {name_r}\n"
                          f"(如果左右 y 接近 → 校正成功)")
        axes[1].axis("off")

        plt.tight_layout()
        out = self.verify_dir / "rectified_pair.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"校正后对比图: {out}")
