import argparse
import os
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------
# Existing rendering function (UNCHANGED)
# ---------------------------------------------------------
def plot_3d_motion(
    kinematic_tree,
    joints,
    title="",
    vbeat=None,
    figsize=(10, 10),
    fps=60,
    radius=4,
    elev=120,
    azim=-90,
    dist=7.5,
    mode="save",                # "show" | "save" | "both"
    save_path="motion.mp4",
    writer="ffmpeg",
    dpi=150,
):

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy()
    frame_number = data.shape[0]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(title, fontsize=16)

    MINS = data.min(axis=(0, 1))
    MAXS = data.max(axis=(0, 1))

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    def init_axes():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.axis("off")

    def update(i):
        ax.cla()
        init_axes()
        ax.view_init(elev=elev, azim=azim)
        ax.dist = dist

        plot_xzPlane(
            MINS[0] - trajec[i, 0],
            MAXS[0] - trajec[i, 0],
            0,
            MINS[2] - trajec[i, 1],
            MAXS[2] - trajec[i, 1]
        )

        if i > 1:
            ax.plot3D(
                trajec[:i, 0] - trajec[i, 0],
                np.zeros_like(trajec[:i, 0]),
                trajec[:i, 1] - trajec[i, 1],
                linewidth=1.0,
                color="blue"
            )

        for j, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            lw = 4.0 if j < 5 else 2.0
            ax.plot3D(
                data[i, chain, 0],
                data[i, chain, 1],
                data[i, chain, 2],
                linewidth=lw,
                color=color
            )

        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_number,
        interval=1000 / fps,
        repeat=False,
        blit=False
    )

    if mode in ("save", "both"):
        ani.save(save_path, fps=fps, writer=writer, dpi=dpi)
        print(f"[OK] Saved: {save_path}")

    if mode in ("show", "both"):
        plt.show()
    else:
        plt.close(fig)

    return ani


# ---------------------------------------------------------
# Kinematic chain (as in your file)
# ---------------------------------------------------------
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20]
]


# ---------------------------------------------------------
# Argparse
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Render motion files")

    parser.add_argument("--motion_dir", type=str, required=True,
                        help="Directory containing .npy motion files")

    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save rendered motions")

    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--mode", type=str, default="save",
                        choices=["save", "show", "both"])

    parser.add_argument("--writer", type=str, default="ffmpeg",
                        choices=["ffmpeg", "pillow"])

    parser.add_argument("--ext", type=str, default="mp4",
                        help="Output format: mp4 or gif")

    return parser.parse_args()


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
def main():
    args = parse_args()

    motion_dir = Path(args.motion_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    motion_files = sorted(motion_dir.glob("*.npy"))
    assert len(motion_files) > 0, "No .npy files found"

    for motion_path in motion_files:
        motion = np.load(motion_path)

        out_path = save_dir / f"{motion_path.stem}.{args.ext}"

        print(f"[INFO] Rendering {motion_path.name}")

        plot_3d_motion(
            KINEMATIC_CHAIN,
            motion,
            title=motion_path.stem,
            fps=args.fps,
            mode=args.mode,
            save_path=str(out_path),
            writer=args.writer,
        )


if __name__ == "__main__":
    main()
