import numpy as np
from scipy.io import loadmat
from scipy.signal import windows


def get_subband_taper(ssl=45, bin_range=(0, 14)):
    win3 = windows.chebwin(3, ssl)
    w1 = win3[0] ** 3
    w2 = win3[0]

    num_bin = bin_range[1] - bin_range[0] + 1
    taper = np.zeros([13, num_bin])
    taper[0] = 1
    taper[1:12:2] = w1
    taper[2:12:2] = w2
    return taper


def get_cvx_tapers():
    mat_file_path_list = [
        r"F:\Projects\PycharmProjects\py_athena_optim\data\input\cm100_13mic_la_taper_x.mat",
        r"F:\Projects\PycharmProjects\py_athena_optim\data\input\cm100_11mic_la_taper_y.mat",
    ]
    tapers = list(map(lambda x: loadmat(x)["w_all"], mat_file_path_list))
    tapers += [get_subband_taper()]
    return tapers


def weight2str(w, num_in_line=8):
    """(num_mic,num_bins) --> (num_mic*num_bins*2)"""
    a = w.reshape(-1).astype(np.float32).astype(str)
    txt = ""
    for i in range(0, len(a), num_in_line):
        txt += "f, ".join(a[i : i + num_in_line]) + "f,"
        if i + num_in_line < len(a):
            txt += "\n"
            txt += "    "
    return txt


def get_line(var_name, taper):
    return f"""static const std::vector<float> {var_name} = {{
    {weight2str(taper)}\n}};\n\n"""


def get_header():
    return "#pragma once\n\n#include <vector>\n\n"


def main():
    out_header_path = "data/output/optim_weights.h"

    tapers = get_cvx_tapers()
    msg = get_header()
    for i, taper in enumerate(tapers):
        msg += get_line(f"g_opt_w{i}", taper)

    with open(out_header_path, "w", encoding="utf8") as fp:
        fp.write(msg)
    print(out_header_path)
    ...


if __name__ == "__main__":
    main()
    ...
