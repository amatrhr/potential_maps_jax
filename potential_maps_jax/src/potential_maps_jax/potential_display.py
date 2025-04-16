"""
core functions and classes  for this potential problem
no constants used
"""
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

GRID_SIZE = 2
GRID = jnp.indices([GRID_SIZE], dtype=jnp.float32)
GRID -= jnp.median(GRID)
PARTICLE_LOCS = ((-1,0), (1,0))
PARTICLE_CHGS = (-1, 1)

def electric_field_of_particle(x0:jnp.ndarray=0, y0:jnp.ndarray=0, charge:float =1., x:jnp.ndarray=1, y:jnp.ndarray=1) -> jnp.ndarray:
    """
    FUnction to calculate the electric field of particle at all points
    :param x0:
    :param y0:
    :param charge:partA

    :return:
    """
    xs = (x-x0).ravel()
    ys = (y-y0).ravel()
    r = jnp.asarray(jnp.meshgrid(xs, ys))
    mags = jnp.apply_along_axis(jnp.linalg.norm, 1, r)
    E = charge * r / mags
    return E

def total_field_at_point(particles, charges):
    """
    Sum over
    :param particles:
    :param charges:
    :return:
    """

def potential_of_particle(particle:jnp.ndarray)->jnp.ndarray:
    V = ...
    return V
#
partA = electric_field_of_particle(x=GRID,y=GRID)
partB = electric_field_of_particle(x=GRID,y=GRID[1])
# gradA = lambda x, y: -1 * potential_of_particle(partA)(x,y)
# VA = lambda x, y: gradA(x,y)[0] + gradA(x,y)[1]

print(partA.squeeze())
print(partB.squeeze())
# print(VA(1.,1.))

