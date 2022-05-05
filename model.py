import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
from matplotlib import animation


class dim3(object):
    def __init__(self, x=1, y=1, z=1):
        self.x = x
        self.y = y
        self.z = z

        self.dims = [x, y, z]


class KernelLauncher(object):
    def __init__(self, grid: dim3, block: dim3):
        self.grid_dim = grid
        self.block_dim = block

    def launch(self, kernel_fn):
        for block_z in range(self.grid_dim.z):
            for block_y in range(self.grid_dim.y):
                for block_x in range(self.grid_dim.x):
                    for thread_z in range(self.block_dim.z):
                        for thread_y in range(self.block_dim.y):
                            for thread_x in range(self.block_dim.x):
                                blockIdx = dim3(block_x, block_y, block_z)
                                threadIdx = dim3(thread_x, thread_y, thread_z)
                                kernel_fn(blockIdx, threadIdx,
                                          self.grid_dim, self.block_dim)

                                # if thread_x % 4 == 0:
                                yield kernel_fn.checkpoint()
                    return


class Kernel(object):
    def __call__(self, block, thread, grid_dim, block_dim):
        pass


if __name__ == '__main__':
    TILE_DIM = 32
    BLOCK_ROWS = 8
    M = 128
    N = 128
    grid_dim = dim3(N//TILE_DIM, M//TILE_DIM)
    block_dim = dim3(TILE_DIM, BLOCK_ROWS)
    launcher = KernelLauncher(grid_dim, block_dim)

    class Task(Kernel):
        def __init__(self):
            self.data = np.zeros(shape=(M, N), dtype='uint8')
            print(self.data)
            self.img = None

        def mark(self, x, y, j):
            if self.data[y+j][x] == 255:
                self.data[y+j][x] = 1
            elif self.data[y+j][x] != 128:
                self.data[y+j][x] = 255

        def __call__(self, blockIdx: dim3, threadIdx: dim3, gridDim: dim3, blockDim: dim3):
            x = blockIdx.x * TILE_DIM + threadIdx.x
            y = blockIdx.y * TILE_DIM + threadIdx.y
            width = gridDim.x * TILE_DIM

            j = 0
            while j < TILE_DIM:
                self.mark(x, y, j)
                j += BLOCK_ROWS

        def checkpoint(self):
            return self.data

    kernel = Task()

    checkpoints = launcher.launch(kernel)

    print(kernel.data)

    fig = plt.figure()
    im = plt.imshow(kernel.data, cmap='gray', vmin=0, vmax=255)

    def init():
        im.set_data(np.zeros((M, N)))

    def animate(i):
        if checkpoints:
            data = next(checkpoints)
            im.set_data(data)
        return im

    # frames = (block_dim.x // 4) * grid_dim.y * grid_dim.x,
    frames = N * M
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames,
                                   interval=6)
    plt.show()
    # with open("myvideo.html", "w") as f:
    #     print(anim.to_html5_video(), file=f)
    # anim.save('1.gif', writer='imagemagick', fps=15)
