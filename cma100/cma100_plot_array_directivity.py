import numpy as np
import plotly.graph_objects as go


def plot_3d_beam_pattern(theta_range, phi_range, power_spec):
    assert power_spec.ndim == 2

    fig = go.Figure(data=[go.Surface(x=theta_range, y=phi_range, z=power_spec)])

    fig.update_traces(
        contours_z=dict(  # 轮廓设置
            show=True,  # 开启是否显示
            usecolormap=True,  # 颜色设置
            highlightcolor="mistyrose",  # 高亮
            project_z=True,
        )
    )

    fig.update_layout(
        title="Beam Pattern",
        autosize=True,
        # scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
        # width=1000,
        # height=700,
        # margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(
            xaxis=dict(title="x: theta"),
            yaxis=dict(title="y: phi"),
            zaxis=dict(title="z: power"),
        ),
    )

    fig.show()
    ...


def cal_beam_pattern_data(coords, freq, angle_delta=1, c=340):
    num_mic = len(coords)

    step = np.deg2rad(angle_delta)
    theta_range = np.arange(0, 2 * np.pi, step).reshape(-1, 1)
    phi_range = np.arange(0, np.pi / 2 + step, step).reshape(1, -1)

    dis = (
        coords[:, 0, None, None] * np.sin(phi_range) * np.cos(theta_range)
        + coords[:, 1, None, None] * np.sin(phi_range) * np.sin(theta_range)
        + coords[:, 2, None, None] * np.cos(phi_range)
    )
    spec_data = np.exp(2j * np.pi * freq * dis / c)

    spec_sum_data = np.sum(spec_data, 0)
    mag_spec = np.abs(spec_sum_data) / num_mic
    power_spec = 20 * np.log10(mag_spec)
    return (
        np.rad2deg(theta_range).reshape(-1),
        np.rad2deg(phi_range).reshape(-1),
        power_spec,
    )


def get_mic_coords(
    coord_z=0,
    delta_theta=np.pi / 3,
    dis2mic0_per_circle=(0.0175, 0.0350, 0.0700, 0.1400, 0.2800),
):
    def keep_in_eps(num, eps=1e-7):
        if abs(num) < eps:
            return 0
        return num

    num_mic = 55
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


if __name__ == "__main__":
    mic_coords, _ = get_mic_coords()
    # PlotUtils.plot_3d_coord(mic_coords)
    # theta_range, phi_range, power_spec = cal_beam_pattern_data(mic_coords, 5000)
    # plot_3d_beam_pattern(theta_range, phi_range, power_spec)

    pick_mic_ids = list(range(19))

    a = mic_coords[pick_mic_ids].T
    for coord in a:
        print(" ".join(coord.astype(str)), end="; ")

    print()
    for _ in range(2):
        print(" ".join(len(pick_mic_ids) * ["0"]), end="; ")
    ...
