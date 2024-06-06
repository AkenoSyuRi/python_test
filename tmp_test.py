import numpy as np
from plot_utils import PlotUtils


def sphere(levels_count=3):
    """This function generates cartesian coordinates (xyz) for a set
    of points forming a 3D sphere. The coordinates are expressed in
    meters and can be used as doas. The result has the format:
    (n_points, 3).

    Arguments
    ---------
    levels_count : int
        A number proportional to the number of points that the user
        wants to generate.
            - If levels_count = 1, then the sphere will have 42 points
            - If levels_count = 2, then the sphere will have 162 points
            - If levels_count = 3, then the sphere will have 642 points
            - If levels_count = 4, then the sphere will have 2562 points
            - If levels_count = 5, then the sphere will have 10242 points
            - ...
        By default, levels_count is set to 4.

    Example
    -------
    # >>> import numpy as np
    # >>> doas = sphere()
    """

    # Generate points at level 0
    h = (5.0**0.5) / 5.0
    r = (2.0 / 5.0) * (5.0**0.5)
    pi = 3.141592654

    pts = np.zeros((12, 3), dtype=np.float32)
    pts[0, :] = np.array([0, 0, 1], dtype=np.float32)
    pts[11, :] = np.array([0, 0, -1], dtype=np.float32)
    pts[1:6, 0] = r * np.sin(2.0 * pi * np.arange(0, 5) / 5.0)
    pts[1:6, 1] = r * np.cos(2.0 * pi * np.arange(0, 5) / 5.0)
    pts[1:6, 2] = h
    pts[6:11, 0] = -r * np.sin(2.0 * pi * np.arange(0, 5) / 5.0)
    pts[6:11, 1] = -r * np.cos(2.0 * pi * np.arange(0, 5) / 5.0)
    pts[6:11, 2] = -h

    # Generate triangles at level 0
    trs = np.zeros((20, 3), dtype=np.int64)

    trs[0, :] = np.array([0, 2, 1], dtype=np.int64)
    trs[1, :] = np.array([0, 3, 2], dtype=np.int64)
    trs[2, :] = np.array([0, 4, 3], dtype=np.int64)
    trs[3, :] = np.array([0, 5, 4], dtype=np.int64)
    trs[4, :] = np.array([0, 1, 5], dtype=np.int64)

    trs[5, :] = np.array([9, 1, 2], dtype=np.int64)
    trs[6, :] = np.array([10, 2, 3], dtype=np.int64)
    trs[7, :] = np.array([6, 3, 4], dtype=np.int64)
    trs[8, :] = np.array([7, 4, 5], dtype=np.int64)
    trs[9, :] = np.array([8, 5, 1], dtype=np.int64)

    trs[10, :] = np.array([4, 7, 6], dtype=np.int64)
    trs[11, :] = np.array([5, 8, 7], dtype=np.int64)
    trs[12, :] = np.array([1, 9, 8], dtype=np.int64)
    trs[13, :] = np.array([2, 10, 9], dtype=np.int64)
    trs[14, :] = np.array([3, 6, 10], dtype=np.int64)

    trs[15, :] = np.array([11, 6, 7], dtype=np.int64)
    trs[16, :] = np.array([11, 7, 8], dtype=np.int64)
    trs[17, :] = np.array([11, 8, 9], dtype=np.int64)
    trs[18, :] = np.array([11, 9, 10], dtype=np.int64)
    trs[19, :] = np.array([11, 10, 6], dtype=np.int64)

    # Generate next levels
    for levels_index in range(levels_count):
        trs_count = trs.shape[0]
        subtrs_count = trs_count * 4

        subtrs = np.zeros((subtrs_count, 6), dtype=np.int64)

        subtrs[0 * trs_count : 1 * trs_count, 0] = trs[:, 0]
        subtrs[0 * trs_count : 1 * trs_count, 1] = trs[:, 0]
        subtrs[0 * trs_count : 1 * trs_count, 2] = trs[:, 0]
        subtrs[0 * trs_count : 1 * trs_count, 3] = trs[:, 1]
        subtrs[0 * trs_count : 1 * trs_count, 4] = trs[:, 2]
        subtrs[0 * trs_count : 1 * trs_count, 5] = trs[:, 0]

        subtrs[1 * trs_count : 2 * trs_count, 0] = trs[:, 0]
        subtrs[1 * trs_count : 2 * trs_count, 1] = trs[:, 1]
        subtrs[1 * trs_count : 2 * trs_count, 2] = trs[:, 1]
        subtrs[1 * trs_count : 2 * trs_count, 3] = trs[:, 1]
        subtrs[1 * trs_count : 2 * trs_count, 4] = trs[:, 1]
        subtrs[1 * trs_count : 2 * trs_count, 5] = trs[:, 2]

        subtrs[2 * trs_count : 3 * trs_count, 0] = trs[:, 2]
        subtrs[2 * trs_count : 3 * trs_count, 1] = trs[:, 0]
        subtrs[2 * trs_count : 3 * trs_count, 2] = trs[:, 1]
        subtrs[2 * trs_count : 3 * trs_count, 3] = trs[:, 2]
        subtrs[2 * trs_count : 3 * trs_count, 4] = trs[:, 2]
        subtrs[2 * trs_count : 3 * trs_count, 5] = trs[:, 2]

        subtrs[3 * trs_count : 4 * trs_count, 0] = trs[:, 0]
        subtrs[3 * trs_count : 4 * trs_count, 1] = trs[:, 1]
        subtrs[3 * trs_count : 4 * trs_count, 2] = trs[:, 1]
        subtrs[3 * trs_count : 4 * trs_count, 3] = trs[:, 2]
        subtrs[3 * trs_count : 4 * trs_count, 4] = trs[:, 2]
        subtrs[3 * trs_count : 4 * trs_count, 5] = trs[:, 0]

        subtrs_flatten = np.concatenate(
            (subtrs[:, [0, 1]], subtrs[:, [2, 3]], subtrs[:, [4, 5]]), axis=0
        )
        subtrs_sorted = np.sort(subtrs_flatten, axis=1)

        index_max = np.max(subtrs_sorted)

        subtrs_scalar = subtrs_sorted[:, 0] * (index_max + 1) + subtrs_sorted[:, 1]

        unique_scalar, unique_indices = np.unique(subtrs_scalar, return_inverse=True)

        unique_values = np.zeros((unique_scalar.shape[0], 2), dtype=unique_scalar.dtype)

        unique_values[:, 0] = np.floor_divide(unique_scalar, index_max + 1)
        unique_values[:, 1] = unique_scalar - unique_values[:, 0] * (index_max + 1)

        trs = np.transpose(np.reshape(unique_indices, (3, -1)))

        pts = pts[unique_values[:, 0], :] + pts[unique_values[:, 1], :]
        pts /= np.repeat(
            np.expand_dims(np.sqrt(np.sum(pts**2, axis=1)), 1), 3, axis=1
        )

    return pts


if __name__ == "__main__":
    pts = sphere(2)
    PlotUtils.plot_3d_coord(pts)
