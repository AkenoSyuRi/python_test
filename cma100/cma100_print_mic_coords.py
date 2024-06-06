import numpy as np


def get_mic_coords(
    coord_z=0,
    delta_theta=np.pi / 3,
    dis2mic0_per_circle=(0.0175, 0.0350, 0.0700, 0.1400, 0.2800, 0.5600),
):
    def keep_in_eps(num, eps=1e-7):
        if abs(num) < eps:
            return 0
        return num

    assert len(dis2mic0_per_circle) > 1

    num_mic = 1 + 6 + 12 * (len(dis2mic0_per_circle) - 1)
    mic_ids = list(range(num_mic))
    mic_coords = [[0, 0, coord_z] for _ in range(num_mic)]

    mic_idx = 1
    num_mic_per_circle = int(2 * np.pi // delta_theta)
    for circle, d in enumerate(dis2mic0_per_circle, 1):
        if circle == 1:
            idx_stride = 1
        else:
            idx_stride = 2

        mic_idx_i0 = -1
        theta = 0
        for i in range(num_mic_per_circle):
            x = keep_in_eps(d * np.cos(theta))
            y = keep_in_eps(d * np.sin(theta))
            mic_coords[mic_idx] = [x, y, coord_z]

            if circle > 1:
                if i > 0:
                    last2th_idx = mic_idx - 2
                    last1th_idx = mic_idx - 1
                    mic_coords[last1th_idx][0] = keep_in_eps(
                        (mic_coords[mic_idx][0] + mic_coords[last2th_idx][0]) / 2
                    )
                    mic_coords[last1th_idx][1] = keep_in_eps(
                        (mic_coords[mic_idx][1] + mic_coords[last2th_idx][1]) / 2
                    )

                    if i == num_mic_per_circle - 1:
                        next1th_idx = mic_idx + 1
                        mic_coords[next1th_idx][0] = keep_in_eps(
                            (mic_coords[mic_idx][0] + mic_coords[mic_idx_i0][0]) / 2
                        )
                        mic_coords[next1th_idx][1] = keep_in_eps(
                            (mic_coords[mic_idx][1] + mic_coords[mic_idx_i0][1]) / 2
                        )
                else:
                    mic_idx_i0 = mic_idx

            mic_idx += idx_stride
            theta += delta_theta
    return np.array(mic_coords), mic_ids


def print_mic_coords_for_matlab(mic_coords):
    coords = list(mic_coords.astype(str))
    print("[")
    for coord in coords:
        print("\t" + " ".join(coord), end=";\n")
    print("];")
    ...


if __name__ == "__main__":
    mic_coords, _ = get_mic_coords()
    # pick_mic_ids = [0, 7, 9, 11, 13, 15, 17]
    # pick_mic_ids = [25, 13, 0, 7, 19]
    # pick_mic_ids = [28, 16, 0, 10, 22]

    # use_mic_coords = mic_coords[pick_mic_ids]
    # print(" ".join(use_mic_coords.reshape(-1).astype(str)))

    print_mic_coords_for_matlab(mic_coords[:55])
    ...
