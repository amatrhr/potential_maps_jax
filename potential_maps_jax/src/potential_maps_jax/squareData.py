"""
Classes and functions to simulate half a square and then
use cosine similarity to fill in the rest. Self-teaching jax/nnx
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import random
from time import time



class SquareData:
    """
    Generate linearly spaced points on the edges of a 2D square
    jitter with uniform noise
    """
    def __init__(self, size:float=6., resolution:int=8, noise_lo:float=0., noise_hi:float=0., seed:int=int(time())):
        self.size = size
        self.resolution = resolution
        self.noise_lo = noise_lo
        self.noise_hi = noise_hi
        self.key = random.key(31212417 + seed)
        self.create_points()


    def create_points(self):
        self.bottom = jnp.column_stack(
            (jnp.linspace(0, self.size, self.resolution, endpoint=True, dtype=jnp.float32),
             jnp.zeros(self.resolution, dtype=jnp.float32)))
        self.top = jnp.column_stack(
            (jnp.linspace(0, self.size, self.resolution, endpoint=True, dtype=jnp.float32),
             self.size*jnp.ones(self.resolution, dtype=jnp.float32)))
        self.left = jnp.column_stack(
            (jnp.zeros(self.resolution, dtype=jnp.float32),
            jnp.linspace(0, self.size, self.resolution, endpoint=True, dtype=jnp.float32)))
        self.right = jnp.column_stack(
            (self.size * jnp.ones(self.resolution, dtype=jnp.float32),
             jnp.linspace(0, self.size, self.resolution, endpoint=True, dtype=jnp.float32)))

        self.points = jnp.vstack((self.bottom, self.left, self.top,  self.right))
        self.jitter = random.uniform(self.key, shape=self.points.shape, minval=self.noise_lo,
                                      maxval=self.noise_hi)
        self.points += self.jitter


    def plot(self, to_save=False, savename="square_display.png"):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
        ax.set_aspect(1)
        ax.scatter(self.points[:,0], self.points[:,1])
        ax.set_axis_off()
        if to_save:
            plt.savefig(savename)

        plt.show()
        plt.close()

    def makedata(self):
        """
        returns a tuple with the first 3/2 * resolution points as the first element (bottom and left sides)
         and the second 3/2 * resolution points as the second element
        :return:
        """
        npoints = self.points.shape[0]//2
        return self.points[0:npoints], self.points[npoints:]



def make_squares_dataset(n_train:int=512, n_test:int=256, noise_lo:float=-2., noise_hi:float=2.,
                 resolution:int=16, size:float=16.)->jnp.array:
        train_data = (SquareData(size=size, resolution=resolution, noise_lo=noise_lo, noise_hi=noise_hi, seed=x+9).makedata()
                      for x in range(n_train))
        test_data = (SquareData(size=size, resolution=resolution, noise_lo=noise_lo, noise_hi=noise_hi, seed=x + 187961).makedata()
                     for x in range(n_test))
        return train_data, test_data


# convert to TF

