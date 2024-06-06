import numpy as np


def get_cma100_mics_coordinates(
    coord_z=3,
    delta_theta=np.pi / 6,
    dis2mic0_per_circle=(0.008, 0.016, 0.032, 0.064, 0.128),
):
    num_mic_per_circle = int(2 * np.pi // delta_theta)  # 12
    all_coords = [[0, 0, coord_z]]
    all_mics = [0] + list(range(1, len(dis2mic0_per_circle) * num_mic_per_circle + 1))

    theta = 0
    for d in dis2mic0_per_circle:
        for _ in range(num_mic_per_circle):
            x = round(d * np.cos(theta), 4)
            y = round(d * np.sin(theta), 4)
            coord = [x, y, coord_z]
            all_coords.append(coord)
            theta += delta_theta

    return all_coords, all_mics


if __name__ == "__main__":
    rou = 1.3
    sample_rate = 16000
    deg = [0, 30, 60, 90, 120, 150]
    theta = np.deg2rad(np.array(deg))
    src_x, src_y, src_z = rou * np.cos(theta), rou * np.sin(theta), 3
    check_mic_ids = list(range(25, 37))
    # check_mic_ids = [45, 46, 47, 48, 57, 58, 59, 60]

    mic_coords, _ = get_cma100_mics_coordinates()

    print("  ", " \t\t".join(map(str, deg)))
    for mic_id in check_mic_ids:
        coord = mic_coords[mic_id]
        d2mic0 = np.sqrt(src_x**2 + src_y**2 + (src_z - 3) ** 2)
        d2mic = np.sqrt(
            np.square(coord[0] - src_x)
            + np.square(coord[1] - src_y)
            + np.square(coord[2] - src_z)
        )
        offset = np.round((d2mic0 - d2mic) / 340 * sample_rate, 4)
        print(mic_id, offset)
    ...
