import jax
from jax import random

from jax import numpy as jnp
import matplotlib.pyplot as plt
from functools import partial


class PotentialDisplay:
    def __init__(self, gridsize: int = 32, numpoints: int = 20, numnegative: int = 1, seed: int = 0,
                 quarternegative=False):
        self.GRID_SIZE = gridsize
        self.GRID_X, self.GRID_Y = jnp.indices([self.GRID_SIZE, self.GRID_SIZE], dtype=jnp.float32)
        self.GRID_X -= jnp.median(self.GRID_X)
        self.GRID_Y -= jnp.median(self.GRID_Y)
        self.FLAT_X = self.GRID_X.flatten()
        self.FLAT_Y = self.GRID_Y.flatten()
        self.COORD_X = jnp.unique(self.FLAT_X)
        self.COORD_Y = jnp.unique(self.FLAT_Y)
        self.NUM_POINTS = numpoints
        self.NUMNEG = numnegative

        key = jax.random.key(31212417 + seed)
        self.X_R = jax.random.choice(key, a=jnp.arange(-self.GRID_SIZE // 2, self.GRID_SIZE // 2, 1), replace=True,
                                     shape=(self.NUM_POINTS,))
        self.Y_R = jax.random.choice(random.key(seed + 1), a=jnp.arange(-self.GRID_SIZE // 2, self.GRID_SIZE // 2, 1),
                                     replace=True, shape=(self.NUM_POINTS,))
        self.PARTICLE_LOCS = jnp.column_stack((self.X_R, self.Y_R))
        self.PARTICLE_CHGS = jnp.ones(self.NUM_POINTS)
        quarter_negative_index = jnp.apply_along_axis(func1d=lambda x: (x[0] > 0) * (x[1] > 0), axis=1, arr=self.PARTICLE_LOCS)
        if numnegative > 0:
            negative_points = jax.random.choice(random.key(seed + 41),
                                                a=jnp.arange(len(self.PARTICLE_CHGS)),
                                                shape=(numnegative,))
            if quarternegative:
                print(self.PARTICLE_CHGS)
                qn_update = jnp.array([negative_points[_] for _, x in enumerate(quarter_negative_index) if x])
                self.PARTICLE_CHGS = self.PARTICLE_CHGS.at[qn_update].set(-1)
                print(self.PARTICLE_CHGS)
                print(self.PARTICLE_LOCS)
            else:
                self.PARTICLE_CHGS = self.PARTICLE_CHGS.at[negative_points].set(-1)

    def electric_field_of_particle(self, x0: jnp.ndarray, x0_charge: jnp.ndarray) -> jnp.ndarray:
        """

        :param x0:
        :param x0_charge:
        :return:
        """

        x_grid = self.GRID_X
        y_grid = self.GRID_Y
        xs = x_grid.flatten() - x0[0]
        ys = y_grid.flatten() - x0[1]

        r = jnp.column_stack((xs, ys))
        # print(f'{r=}')
        mags = xs ** 2 + ys ** 2
        # print(f'{mags=}')
        e_field = x0_charge * r / jnp.expand_dims(mags, 1)
        return e_field

    def total_field_at_point(self):

        """
        Sum over
        :param particles:
        :param charges:
        :return:
        """
        total_field = 0
        for idx, particle_loc in enumerate(self.PARTICLE_LOCS):
            total_field += self.electric_field_of_particle(x0=particle_loc, x0_charge=self.PARTICLE_CHGS[idx])
        return total_field

    def potential_of_particle(self, x0: jnp.ndarray, x0_charge: jnp.ndarray) -> jnp.ndarray:
        """
        FUnction to calculate the electric field of particle at all points
        :param x0:
        :param y0:
        :param charge:partA

        :return:
        """
        x_grid = self.GRID_X
        y_grid = self.GRID_Y
        xs = x_grid - x0[0]
        ys = y_grid - x0[1]

        # print(f'{r=}')
        mags = jnp.sqrt(xs ** 2 + ys ** 2)
        # print(f'{mags=}')
        potential = x0_charge * (-1 / mags)
        return potential

    #

    def potential_at_point(self):

        """
        Sum over
        :param particles:
        :param charges:
        :return:
        """
        total_potential = 0
        for idx, particle_loc in enumerate(self.PARTICLE_LOCS):
            total_potential += self.potential_of_particle(x0=particle_loc,
                                                          x0_charge=self.PARTICLE_CHGS[idx])
        return total_potential

    def plot(self, contour=True, to_save=False, savename="potential_display.png"):
        fig, ax = plt.subplots(figsize=(16, 9), dpi=270)
        ax.set_aspect(1)

        partC = self.total_field_at_point()
        partD = self.potential_at_point()

        if contour:
            ctr0 = ax.contourf(self.COORD_X, self.COORD_Y, partD.T)
            plt.colorbar(ctr0, ax=ax)

        for _, j in enumerate(self.FLAT_X):
            ax.arrow(x=j, y=self.FLAT_Y[_], dx=partC[_, 0], dy=partC[_, 1])

        if to_save:
            plt.savefig(savename)

        plt.show()
        plt.close()


pd2 = PotentialDisplay(gridsize=32, numpoints=24, numnegative=8,quarternegative=False,seed=141)
pd2.plot(to_save=True)
pd2 = PotentialDisplay(gridsize=32, numpoints=24, numnegative=8,quarternegative=True,seed=141)
pd2.plot(to_save=True, savename="qn_display.png")
