import jax
from jax import random
key = random.key(31212417)

from jax import numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

GRID_SIZE = 256
GRID_X, GRID_Y = jnp.indices([GRID_SIZE, GRID_SIZE], dtype=jnp.float32)
GRID_X -= jnp.median(GRID_X)
GRID_Y-= jnp.median(GRID_Y)
COORD_X = jnp.unique(GRID_X.flatten())
COORD_Y = jnp.unique(GRID_Y.flatten())
NUM_POINTS = 512

# PARTICLE_LOCS = jnp.asarray(((-7,0), (7,0), (3.5,-4), (6.1,6)))
X_R = jax.random.choice(key, a = jnp.arange(-GRID_SIZE//2,GRID_SIZE//2,1),replace=True, shape=(NUM_POINTS,))
Y_R = jax.random.choice( random.key(2132), a = jnp.arange(-GRID_SIZE//2,GRID_SIZE//2,1),replace=True, shape=(NUM_POINTS,))
PARTICLE_LOCS = jnp.column_stack((X_R, Y_R))
PARTICLE_CHGS = jax.random.choice( random.key(33315),\
                                  a = jnp.asarray((-1,1)),
                                  replace=True, shape=(NUM_POINTS,))

def electric_field_of_particle(x0:jnp.ndarray=PARTICLE_LOCS[0],
                               x0_charge:float =PARTICLE_CHGS[0],
                               x_grid:jnp.ndarray=GRID_X,
                               y_grid:jnp.ndarray=GRID_Y) -> jnp.ndarray:
    """
    FUnction to calculate the electric field of particle at all points
    :param x0:
    :param y0:
    :param charge:partA

    :return:
    """
    xs = x_grid.flatten() - x0[0]
    ys = y_grid.flatten() - x0[1]

    r = jnp.column_stack((xs,ys))
    # print(f'{r=}')
    mags = xs**2 + ys**2
    # print(f'{mags=}')
    E = x0_charge * r / jnp.expand_dims(mags, 1)
    return E

def total_field_at_point(particles, charges, x_grid,y_grid):

    """
    Sum over
    :param particles:
    :param charges:
    :return:
    """
    total_field = 0
    for idx, particle_loc in enumerate(particles):
      total_field += electric_field_of_particle(x0=particle_loc,
                                                x0_charge=charges[idx],
                                   x_grid=x_grid, y_grid=y_grid)
    return total_field

def potential_of_particle(x0:jnp.ndarray=PARTICLE_LOCS[0],
                               x0_charge:float =PARTICLE_CHGS[0],
                               x_grid:jnp.ndarray=GRID_X,
                               y_grid:jnp.ndarray=GRID_Y) -> jnp.ndarray:
    """
    FUnction to calculate the electric field of particle at all points
    :param x0:
    :param y0:
    :param charge:partA

    :return:
    """
    xs = x_grid - x0[0]
    ys = y_grid - x0[1]

    
    # print(f'{r=}')
    mags = jnp.sqrt(xs**2 + ys**2)
    # print(f'{mags=}')
    V = x0_charge * (1 / mags)
    return V
#

def potential_at_point(particles, charges, x_grid, y_grid):

    """
    Sum over
    :param particles:
    :param charges:
    :return:
    """
    total_potential = 0
    for idx, particle_loc in enumerate(particles):
      total_potential += potential_of_particle(x0=particle_loc,
                                                x0_charge=charges[idx],
                                   x_grid=x_grid, y_grid=y_grid)
    return total_potential
