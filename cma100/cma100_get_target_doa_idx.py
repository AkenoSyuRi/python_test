import math
from typing import Dict

import numpy as np
from scipy.spatial import KDTree

g_positive_hemisphere: bool = ...


class Coord:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Coord(self.x + other.x, self.y + other.y, self.z + other.z)

    def __truediv__(self, scalar: float):
        if scalar == 0:
            raise ValueError("Division by zero")
        return Coord(self.x / scalar, self.y / scalar, self.z / scalar)

    def normalized(self):
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if length == 0:
            return Coord(0, 0, 0)
        return self / length

    def __repr__(self):
        return f"Coord({self.x}, {self.y}, {self.z})"


class Triangle:
    def __init__(self, a: int, b: int, c: int):
        self.a = a
        self.b = b
        self.c = c

    def __repr__(self):
        return f"Triangle({self.a}, {self.b}, {self.c})"


def combine_indices(a: int, b: int) -> int:
    if a > b:
        a, b = b, a
    return (a << 32) | b


def cal_sphere_coords(levels_count: int, positive_hemisphere: bool):
    h = math.sqrt(5.0) / 5.0
    r = (2.0 / 5.0) * math.sqrt(5.0)

    # Initial points
    pts = [...] * 12
    pts[0] = Coord(0, 0, 1)
    pts[-1] = Coord(0, 0, -1)
    for i in range(5):
        angle = 2.0 * np.pi * i / 5.0
        pts[i + 1] = Coord(r * math.sin(angle), r * math.cos(angle), h)
        pts[i + 6] = Coord(-r * math.sin(angle), -r * math.cos(angle), -h)

    # Initial triangles
    trs = [
        Triangle(0, 2, 1),
        Triangle(0, 3, 2),
        Triangle(0, 4, 3),
        Triangle(0, 5, 4),
        Triangle(0, 1, 5),
        Triangle(9, 1, 2),
        Triangle(10, 2, 3),
        Triangle(6, 3, 4),
        Triangle(7, 4, 5),
        Triangle(8, 5, 1),
        Triangle(4, 7, 6),
        Triangle(5, 8, 7),
        Triangle(1, 9, 8),
        Triangle(2, 10, 9),
        Triangle(3, 6, 10),
        Triangle(11, 6, 7),
        Triangle(11, 7, 8),
        Triangle(11, 8, 9),
        Triangle(11, 9, 10),
        Triangle(11, 10, 6),
    ]

    for _ in range(levels_count):
        subtrs = []
        midpoint_cache: Dict[int, int] = {}
        new_pts = pts[:]
        new_pts_count = len(pts)

        def get_midpoint(a: int, b: int) -> int:
            nonlocal new_pts_count
            key = combine_indices(a, b)
            if key not in midpoint_cache:
                midpoint = (pts[a] + pts[b]) / 2.0
                midpoint = midpoint.normalized()
                new_pts.append(midpoint)
                midpoint_cache[key] = new_pts_count
                new_pts_count += 1
            return midpoint_cache[key]

        for tri in trs:
            a, b, c = tri.a, tri.b, tri.c
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            subtrs.append(Triangle(a, ab, ca))
            subtrs.append(Triangle(b, bc, ab))
            subtrs.append(Triangle(c, ca, bc))
            subtrs.append(Triangle(ab, bc, ca))

        trs = subtrs
        pts = new_pts

    result = [
        [pt.x, pt.y, pt.z]
        for pt in pts
        if (positive_hemisphere and pt.z >= 0)
        or (not positive_hemisphere and pt.z <= 0)
    ]
    return np.array(result)


def direction2coord(az, el):
    if el == 0:
        x = 0
        y = 0
        z = -1
    else:
        az_rad = np.deg2rad(az)
        el_rad = np.deg2rad(el)
        x = np.cos(az_rad)
        y = np.sin(az_rad)
        z = np.cos(el_rad)

    if g_positive_hemisphere:
        z = -z

    coord = np.array([x, y, z])
    return coord


def main():
    global g_positive_hemisphere
    g_positive_hemisphere = bool(1)

    doa_coords = cal_sphere_coords(3, g_positive_hemisphere)
    tree = KDTree(doa_coords)

    # PlotUtils.plot_3d_coord(doa_coords)

    idx2 = tree.query(direction2coord(270, 81.5))[1]
    print(g_positive_hemisphere, idx2, doa_coords[idx2])
    ...


if __name__ == "__main__":
    main()
    ...
