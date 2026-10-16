"""
Replay my 12 hand demos on the LIBERO Panda, compartment included.

Keep from the human: motion shape, timing, grasp/release moments, chosen compartment.
Take from the simulator: where the object and the basket actually are.

  1. Load each trajectory (table frame: right, far, up) with its grasp/release events
     and the release point measured on the board (ray-table intersection).
  2. Compartment mapping: release point as a fraction of the board's half-size ->
     the same fraction of the basket's interior (my far-left = basket's far-left).
  3. Keyframe-anchored retargeting: a piecewise-linear offset makes the path start at
     the robot's pose, hit the object at the grasp moment and the mapped compartment at
     the release moment; the shape of my motion in between is kept.
  4. Track the path with LIBERO's end-effector delta controller; gripper from my pinch.
  5. Log: LIBERO success, compartment reached in the basket vs intended, landing error,
     failure type.

Usage (Colab, from the repo root):
  python scripts/retarget_replay.py --n_init 5 --save_videos 1
Outputs: results/sim/replay_results.csv, results/sim/summary.png, results/sim/videos/*.mp4
"""
import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Table frame (right, far, up) -> simulator world (x forward from the robot, y left, z up).
# The camera side of the board is mapped to the robot's side of the table.
R_TABLE_TO_SIM = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
ZONES = ["Morning", "Noon", "Afternoon", "Night"]
CONTROL_HZ, POS_SCALE, ROT_SCALE = 20, 0.05, 0.5
GRIP_OPEN, GRIP_CLOSED = -1.0, 1.0


def zone_of(right, far):
    if far >= 0:
        return "Morning" if right < 0 else "Noon"
    return "Afternoon" if right < 0 else "Night"


# ----------------------------------------------------------------------------- human side
def load_demo(path):
    d = np.load(path, allow_pickle=True)
    i_g, i_r = int(d["i_grasp"]), int(d["i_release"])
    if i_g < 0 or i_r <= i_g:
        raise ValueError("no grasp/release detected")
    hr, hf = d["half_extents"]
    rel = d["release"]
    frac = np.array([rel[0] / hr, rel[1] / hf])            # release as a fraction of the board
    return dict(t=d["t"], pos=d["pos"] @ R_TABLE_TO_SIM.T, i_g=i_g, i_r=i_r,
                intended=str(d["intended"]), perceived=str(d["perceived"]), frac=np.clip(frac, -1, 1))


def retarget(t, pos, i_g, i_r, start, grasp_tgt, release_tgt, approach_window=0.5):
    """Piecewise-linear offset in time so the path passes through the three anchors."""
    n = len(t)
    anchors = [0, i_g, i_r, n - 1]
    offs = np.array([start - pos[0], grasp_tgt - pos[i_g], release_tgt - pos[i_r], release_tgt - pos[i_r]])
    path = pos + np.stack([np.interp(np.arange(n), anchors, offs[:, k]) for k in range(3)], axis=1)
    # Last part of the approach comes straight down onto the object (a parallel gripper needs it)
    idx = np.flatnonzero((t > t[i_g] - approach_window) & (np.arange(n) <= i_g))
    if len(idx) > 1:
        w = np.linspace(0, 1, len(idx))[:, None]
        path[idx, :2] = (1 - w) * path[idx, :2] + w * grasp_tgt[:2]
    path[:, 2] = np.maximum(path[:, 2], grasp_tgt[2])      # never below the grasp height
    return path


def build_plan(t, path, i_g, i_r, hold_s=0.5):
    """Resample to the control rate and hold briefly at grasp and release so the gripper can act."""
    t_ctrl = np.arange(0, t[-1], 1.0 / CONTROL_HZ)
    targets = np.stack([np.interp(t_ctrl, t, path[:, k]) for k in range(3)], axis=1)
    grip = np.where((t_ctrl >= t[i_g]) & (t_ctrl <= t[i_r]), GRIP_CLOSED, GRIP_OPEN)
    hold = int(hold_s * CONTROL_HZ)
    for t_ev, g in ((t[i_r], GRIP_OPEN), (t[i_g], GRIP_CLOSED)):
        k = min(int(np.searchsorted(t_ctrl, t_ev)), len(t_ctrl) - 1)
        targets = np.insert(targets, k, np.repeat(targets[k:k + 1], hold, axis=0), axis=0)
        grip = np.insert(grip, k, np.full(hold, g))
    return targets, grip


# ----------------------------------------------------------------------------- simulator side
def make_env(obj, img_size):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    names = [suite.get_task(i).language for i in range(suite.n_tasks)]
    matches = [i for i, s in enumerate(names) if obj.replace("_", " ") in s]
    if not matches:
        raise SystemExit("No LIBERO-Object task mentions '%s'. Tasks:\n  %s" % (obj, "\n  ".join(names)))
    task = suite.get_task(matches[0])
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=img_size, camera_widths=img_size)
    env.seed(0)
    return env, task.language, suite.get_task_init_states(matches[0])


def find_key(obs, name):
    keys = [k for k in obs if k.endswith("_pos") and name.replace(" ", "_") in k]
    if not keys:
        raise KeyError(f"No '{name}' in observations: {sorted(k for k in obs if k.endswith('_pos'))}")
    return keys[0]


def reset_scene(env, init_state, settle=10):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(settle):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, GRIP_OPEN])
    return obs


def rollout(env, init_state, targets, grip, obj_key, basket_key, record, tail=40):
    obs = reset_scene(env, init_state)
    rot0 = Rotation.from_quat(obs["robot0_eef_quat"])
    z0 = obs[obj_key][2]
    frames, errs, max_lift, success = [], [], 0.0, False
    for target, g in list(zip(targets, grip)) + [(targets[-1], GRIP_OPEN)] * tail:
        eef = obs["robot0_eef_pos"]
        rot_err = (rot0 * Rotation.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec()
        action = np.concatenate([np.clip((target - eef) / POS_SCALE, -1, 1),
                                 np.clip(rot_err / ROT_SCALE, -1, 1), [g]])
        obs, _, done, _ = env.step(action.tolist())
        errs.append(np.linalg.norm(target - eef))
        max_lift = max(max_lift, obs[obj_key][2] - z0)
        if record:
            frames.append(obs["agentview_image"][::-1, ::-1])      # LIBERO renders upside down
        success = success or bool(done) or env.check_success()
    return success, obs[obj_key].copy(), obs[basket_key].copy(), max_lift, float(np.mean(errs)), frames


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default="trajectories/*.npz")
    ap.add_argument("--obj", default="cream_cheese", help="LIBERO-Object item standing in for the medicine box")
    ap.add_argument("--n_init", type=int, default=5, help="LIBERO scene layouts per demo")
    ap.add_argument("--basket_half", type=float, nargs=2, default=[0.04, 0.04],
                    help="half-size of the basket interior used for compartments (x, y) [m]")
    ap.add_argument("--grasp_dz", type=float, default=0.0)
    ap.add_argument("--place_dz", type=float, default=0.10, help="release height above the basket origin [m]")
    ap.add_argument("--save_videos", type=int, default=1, help="videos per demo")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--out", default="results/sim")
    args = ap.parse_args()

    out = Path(args.out); (out / "videos").mkdir(parents=True, exist_ok=True)
    env, language, init_states = make_env(args.obj, args.img_size)
    print(f"LIBERO task: '{language}'")
    bx, by = args.basket_half

    rows = []
    for path in sorted(glob.glob(args.traj)):
        name = Path(path).stem
        try:
            demo = load_demo(path)
        except ValueError as e:
            print(f"skip {name}: {e}"); continue
        for k in range(args.n_init):
            obs = reset_scene(env, init_states[k])
            obj_key, basket_key = find_key(obs, args.obj), find_key(obs, "basket")
            start, obj, basket = obs["robot0_eef_pos"].copy(), obs[obj_key].copy(), obs[basket_key].copy()
            # My compartment choice, as a fraction of the board, mapped into the basket
            cell = R_TABLE_TO_SIM @ np.array([demo["frac"][0] * by, demo["frac"][1] * bx, 0.0])
            release_tgt = basket + cell + np.array([0, 0, args.place_dz])
            grasp_tgt = obj + np.array([0, 0, args.grasp_dz])
            path_sim = retarget(demo["t"], demo["pos"], demo["i_g"], demo["i_r"], start, grasp_tgt, release_tgt)
            targets, grip = build_plan(demo["t"], path_sim, demo["i_g"], demo["i_r"])
            record = k < args.save_videos
            success, obj_end, basket_end, lift, err, frames = rollout(
                env, init_states[k], targets, grip, obj_key, basket_key, record)

            # Where did it land, in basket compartments? (sim x = far, sim y = left)
            d = obj_end - basket_end
            reached = zone_of(-d[1], d[0])
            target_rel = cell[:2]
            landing_err = 100 * np.linalg.norm(d[:2] - target_rel)
            if lift < 0.03:
                outcome = "missed grasp"
            elif not success:
                outcome = "dropped / off target"
            elif reached != demo["intended"]:
                outcome = "wrong compartment"
            else:
                outcome = "correct compartment"
            rows.append(dict(demo=name, init=k, intended=demo["intended"], perceived=demo["perceived"],
                             in_basket=int(success), reached=reached,
                             correct_compartment=int(success and reached == demo["intended"]),
                             outcome=outcome, landing_err_cm=round(landing_err, 1),
                             lift_cm=round(100 * lift, 1), track_err_cm=round(100 * err, 1)))
            print(f"{name:13s} layout {k}: {outcome:21s} reached {reached:9s} landing err {landing_err:4.1f} cm")
            if record and frames:
                import imageio
                imageio.mimsave(out / "videos" / f"{name}_layout{k}.mp4", frames, fps=CONTROL_HZ)

    with open(out / "replay_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary_figure(rows, out / "summary.png")

    n = len(rows)
    print(f"\n{n} rollouts | box in basket {sum(r['in_basket'] for r in rows)}/{n} | "
          f"intended compartment {sum(r['correct_compartment'] for r in rows)}/{n}")
    for o in ["correct compartment", "wrong compartment", "dropped / off target", "missed grasp"]:
        print(f"  {o:21s} {sum(r['outcome'] == o for r in rows)}")


def summary_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outcomes = ["correct compartment", "wrong compartment", "dropped / off target", "missed grasp"]
    colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bottom = np.zeros(len(ZONES))
    for o, c in zip(outcomes, colors):
        vals = np.array([sum(r["intended"] == z and r["outcome"] == o for r in rows) for z in ZONES])
        ax[0].bar(ZONES, vals, bottom=bottom, color=c, label=o); bottom += vals
    ax[0].set_ylabel("rollouts"); ax[0].set_title("Outcome per intended compartment"); ax[0].legend(fontsize=8)
    errs = np.sort([r["landing_err_cm"] for r in rows if r["in_basket"]])
    if len(errs):
        ax[1].plot(errs, np.arange(1, len(errs) + 1) / len(rows), drawstyle="steps-post")
    ax[1].set_xlabel("compartment radius [cm]"); ax[1].set_ylabel("fraction of rollouts landing inside")
    ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.3)
    ax[1].set_title("How small could a compartment be?")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()