import numpy as np
from vertfrac.instances import label_instances


def _chain(n=5, neck=6, gap=0, arch=False):
    m = np.zeros((60, 60, 40 + n * 26), np.uint8)
    for k in range(n):
        z0 = 8 + k * 26
        m[12:36, 14:38, z0:z0 + 21] = 1                        # vertebral body
        if arch:
            m[22:34, 40:52, z0 + 2:z0 + 19] = 1                # posterior element, joined to the body (same height)
            m[24:32, 36:42, z0 + 6:z0 + 14] = 1
        if k < n - 1 and gap == 0:
            c = 25 - neck // 2
            m[c:c + neck, c + 2:c + 2 + neck, z0 + 21:z0 + 26] = 1       # thin disc-like connection
    return m


def test_numbered_top_down_and_count():
    inst = label_instances(_chain())
    assert inst.max() == 5
    zs = [np.argwhere(inst == i)[:, 2].mean() for i in range(1, 6)]
    assert zs == sorted(zs, reverse=True)


def test_separated_vertebrae():
    assert label_instances(_chain(gap=1)).max() == 5


def test_posterior_elements_do_not_create_extra_vertebrae():
    assert label_instances(_chain(arch=True)).max() == 5


def test_empty_mask():
    assert label_instances(np.zeros((10, 10, 10))).max() == 0
