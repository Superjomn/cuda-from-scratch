import re
import subprocess

import matplotlib.pyplot as plt

kernels = {
    0: ('smem_naive', ('blue', '-')),
    1: ("reduce_smem_1_avoid_divergent_warps", ('blue', '-.')),
    3: ("reduce_smem_3_read_two", ('blue', '^')),
    10: ("reduce_warp_shlf", ('red', '-')),
    11: ("reduce_warp_shlf_read_2", ('red', '-.')),
    12: ("reduce_warp_shlf_read_4", ('red', '^')),
    20: ('atomic_naive', ('green', '-')),
    70: ("reduce_warp_shlf_read_1_atomic", ('green', '-.')),
    71: ("reduce_warp_shlf_read_2_atomic", ('green', 'v')),
    73: ("reduce_warp_shlf_read_8_atomic", ('green', '^')),
}


def run_profile(kernel: int, n: int):
    cmd = [
        "./2-reduce",
        "-profile",
        "-block_size",
        "256",
        "-n",
        str(n),
        f"-kernel",
        f"{kernel}",
    ]
    result = subprocess.run(cmd,
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    output = result.stderr.decode()

    match = re.search(r'bandwidth: (\d+\.\d+)', output)
    if match:
        return float(match.group(1))


def batch_profile(kernel: int):
    n = 1024
    for i in range(16):
        yield n, run_profile(kernel, n)
        n *= 2


def profile():

    def get_data():
        for kernel, (kernel_name, style) in kernels.items():
            data = [kernel_name, [], style]

            for n, bandwidth in batch_profile(kernel):
                data[1].append((n, bandwidth))
            yield data

    plt.figure(dpi=300, figsize=(10, 6))

    for kernel, values, style in get_data():
        x, y = zip(*values)
        kwargs = {}
        if style[1] in ['-', '-.', '--', ':']:
            kwargs['linestyle'] = style[1]
        else:
            kwargs['marker'] = style[1]

        plt.plot(x, y, label=kernel, color=style[0], **kwargs)

    plt.legend()
    plt.show()
    plt.ylabel("Bandwidth (GB/s)")
    plt.xlabel("n")
    plt.savefig("profile-reduce.png")


profile()
