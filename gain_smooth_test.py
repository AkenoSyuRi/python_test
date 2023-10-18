import numpy as np
from icecream import ic


def gain_smooth(gain, lam=0.6):
    expected_gain = lam * gain[..., :-1]
    actual_gain = gain[..., 1:]

    flags = expected_gain > actual_gain
    actual_gain[flags] = expected_gain[flags]
    ...


if __name__ == "__main__":
    np.random.seed(1)
    lam = 0.6

    a = np.random.rand(20)
    ic(a)
    expected_gain = lam * a[:-1]
    actual_gain = a[1:]

    flag = expected_gain - actual_gain > 0
    ic(flag, len(flag))

    ic(expected_gain[flag])
    ic(actual_gain[flag])

    actual_gain[flag] = expected_gain[flag]

    ic(a)
    ...
