import numpy as np


def calc_direction(k_Dir, directions):
    Direction_vals = []
    Direction_vals.append(
        np.where(directions == 0, 1, np.where(directions == 1, -1, 0))
    )
    Direction_vals.append(
        np.where(directions == 1, 1, np.where(directions == 0, -1, 0))
    )
    Direction_vals.append(
        np.where(directions == 2, 1, np.where(directions == 3, -1, 0))
    )
    Direction_vals.append(
        np.where(directions == 3, 1, np.where(directions == 2, -1, 0))
    )
    Direction_vals.append(np.zeros_like(directions))
    return np.exp(k_Dir * Direction_vals)


def calc_stress(k_Str, stresses):
    return np.exp(-k_Str * (stresses / 20))


def p_ij(
    Map,
    positions,
    SFF,
    DFF,
    params=[3, 1],
    d="Neumann",
):
    k_S = params[0]
    k_D = params[1]
    shift_axis = [(-1, 1), (1, 0), (1, 1), (-1, 0), (0, 0)]  # 右上左下止
    if d != "Moore":
        M_vals, S_vals, D_vals = (
            np.empty((5, len(positions))),
            np.empty((5, len(positions))),
            np.empty((5, len(positions))),
        )
    else:
        M_vals, S_vals, D_vals = (
            np.empty((9, len(positions))),
            np.empty((9, len(positions))),
            np.empty((9, len(positions))),
        )

    sh = 0
    for idx, (shift, axis) in enumerate(shift_axis):
        M_shift = np.roll(Map, shift=shift, axis=axis)
        # M_vals.append(M_shift[positions[:, 0], positions[:, 1]])
        M_vals[sh] = M_shift[positions[:, 0], positions[:, 1]]
        S_shift = np.roll(SFF, shift=shift, axis=axis)
        # S_vals.append(S_shift[positions[:, 0], positions[:, 1]])
        S_vals[sh] = S_shift[positions[:, 0], positions[:, 1]]
        D_shift = np.roll(DFF, shift=shift, axis=axis)
        # D_vals.append(D_shift[positions[:, 0], positions[:, 1]])
        D_vals[sh] = D_shift[positions[:, 0], positions[:, 1]]
        sh += 1
        if (d == "Moore") & (idx != 4):
            shift, axis = shift_axis[:4][idx - 1]
            M_shift = np.roll(M_shift, shift=shift, axis=axis)
            # M_vals.append(M_shift[positions[:, 0], positions[:, 1]])
            M_vals[sh] = M_shift[positions[:, 0], positions[:, 1]]
            S_shift = np.roll(S_shift, shift=shift, axis=axis)
            # S_vals.append(S_shift[positions[:, 0], positions[:, 1]])
            S_vals[sh] = S_shift[positions[:, 0], positions[:, 1]]
            D_shift = np.roll(D_shift, shift=shift, axis=axis)
            # D_vals.append(D_shift[positions[:, 0], positions[:, 1]])
            D_vals[sh] = D_shift[positions[:, 0], positions[:, 1]]
            sh += 1

    S_max = S_vals.max(axis=0)
    S_vals = np.where(S_vals == -1, S_max, S_vals)
    S_vals = S_vals - S_vals.min(axis=0)

    # 各方向の移動確率を計算
    probs = np.exp(-k_S * S_vals)
    probs *= np.exp(k_D * D_vals)

    probs *= M_vals != 2
    probs[:-1] *= M_vals[:-1] != 1

    if d == "Neumann":
        move = [(0, 1), (-1, 0), (0, -1), (1, 0), (0, 0)]  # 右上左下止
    else:
        move = [
            (0, 1),
            (1, 1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (-1, -1),
            (1, 0),
            (1, -1),
            (0, 0),
        ]  # 右右下上右上左左上下左下止
    # for idx, pos in enumerate(positions):
    #     for i, dir in enumerate(move):
    #         next_pos = tuple(np.array(pos) + np.array(dir))
    #         if Map[next_pos] == 3:
    #             probs[:, idx] = 0
    #             probs[i, idx] = 1
    #             break
    move = np.array(move)
    p = positions[np.newaxis, :, :] + move[:, np.newaxis, :]
    Map_exit = Map[p[..., 0], p[..., 1]]
    is_three = Map_exit == 3
    has_three = np.any(is_three, axis=0)
    probs_updated = np.where(is_three, 1, 0)
    probs_updated *= np.cumsum(probs_updated, axis=0) == 1
    probs = np.where(has_three, probs_updated, probs)

    sums = np.sum(probs, axis=0)
    sums[sums == 0] = 1

    probs = probs / sums
    probs_tmp = 1 - probs[:-1, :].sum(axis=0)
    probs_tmp[probs_tmp < 0] = 0
    probs[-1, :] = probs_tmp
    return probs
