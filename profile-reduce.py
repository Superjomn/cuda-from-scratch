import re
import subprocess

import matplotlib.pyplot as plt

kernels = {
    0: 'smem_naive',
    20: 'atomic_naive',
    1: "reduce_smem_1_avoid_divergent_warps",
    11: "reduce_warp_shlf_read_2",
    12: "reduce_warp_shlf_read_4",
    71: "reduce_warp_shlf_read_2_atomic",
    73: "reduce_warp_shlf_read_8_atomic",
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
        for kernel, kernel_name in kernels.items():
            data = [kernel_name, []]

            for n, bandwidth in batch_profile(kernel):
                data[1].append((n, bandwidth))
            yield data

    plt.figure(dpi=300)

    for kernel, values in get_data():
        x, y = zip(*values)
        plt.plot(x, y, label=kernel)

    plt.legend()
    plt.show()
    plt.savefig("profile-reduce.png")


profile()
