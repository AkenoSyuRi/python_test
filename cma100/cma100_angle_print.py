import re

import matplotlib.pyplot as plt
import numpy as np
from adbutils import adb

plt.figure(figsize=(12, 5))


class Cma100Helper:
    def __init__(self, local_exe_path, force_push=True):
        self._dev = adb.device()
        self._exe = "/tmp/cma100"
        self._force_push = force_push
        self._local_exe_path = local_exe_path
        self._ang_pat = re.compile(r"theta: (\d{3}), phi: (\d{2})")
        assert self._check_exe_exists(), "you must push the cma100 to the /tmp dir"

    def _check_exe_exists(self):
        ret = self._dev.shell(f"if [ -f {self._exe} ]; then echo 1; else echo 0; fi")
        exists = bool(ret)

        if self._force_push or not exists:
            self._dev.push(self._local_exe_path, self._exe)
        self._dev.shell(f"chmod +x {self._exe}")

        return exists

    def read_doa_info(self, buff_len=256):
        with self._dev.shell(self._exe, stream=True) as stream:
            while not stream.closed:
                res = stream.read_string(buff_len)
                print(res)

                m = self._ang_pat.search(res)
                if m:
                    self.plot_angle(int(m.group(1)), int(m.group(2)))

    @staticmethod
    def plot_angle(theta, phi):
        plt.clf()

        plt.subplot(121, projection="polar")
        plt.title(f"theta: {theta:03d}")
        plt.polar([0, np.deg2rad(theta)], [0, 1], "ro-", lw=2)
        plt.ylim(0, 1)  # 设置极轴的上下限

        plt.subplot(122, projection="polar")
        plt.title(f"phi: {phi:02d}")
        plt.polar([0, np.deg2rad(phi)], [0, 1], "ro-", lw=2)
        plt.ylim(0, 1)  # 设置极轴的上下限

        plt.pause(0.1)
        ...


if __name__ == "__main__":
    local_exe_path = r"D:\Temp\cma100"

    cma100 = Cma100Helper(local_exe_path)
    cma100.read_doa_info()
    ...
