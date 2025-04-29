import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from potential_display import PotentialDisplay


class QuarterNegative:
    def __init__(self, gridsize: int = 32, numpoints: int = 20, numnegative: int = 1, seed:int = 0):
        self.bigSquare = PotentialDisplay(gridsize=gridsize, numpoints=numpoints, numnegative = 0, seed =seed)
        self.smallSquare = PotentialDisplay(gridsize=gridsize//4, numpoints=numpoints, numnegative=numnegative, seed=seed+3)

    def combine_systems(self):

        # pad the systems (this should be a separate function)
        self.bigE = self.bigSquare.total_field_at_point()
        smallE = self.smallSquare.total_field_at_point()
        Erows = (self.bigE.shape[0]//2 - smallE.shape[0]//2, self.bigE.shape[0]//2 + smallE.shape[0]//2)
        Ecalc = self.bigE[Erows[0]:Erows[1],:] + smallE

        self.bigV = self.bigSquare.potential_at_point()
        smallV = self.smallSquare.potential_at_point()
        Vpad = self.bigV.shape[0] - smallV.shape[0]
        # breakpoint()
        self.bigE = self.bigE.at[Erows[0]:Erows[1],:].set(Ecalc)
        self.e_field = self.bigE
        self.potential = self.bigV + jnp.pad(smallV, pad_width=((Vpad//2,Vpad//2),(Vpad//2,Vpad//2)))

    def plot(self, contour=True, to_save=False):
        fig, ax = plt.subplots(figsize=(16, 9), dpi=270)
        ax.set_aspect(1)

        if contour:
            ctr0 = ax.contourf(self.bigSquare.COORD_X, self.bigSquare.COORD_Y, self.potential.T)
            plt.colorbar(ctr0, ax=ax)

        for _, j in enumerate(self.bigSquare.FLAT_X):
            ax.arrow(x=j, y=self.bigSquare.FLAT_Y[_], dx=self.e_field[_, 0], dy=self.e_field[_, 1])

        if to_save:
            plt.savefig("combined_system_EV.png")

        plt.show()
        plt.close()



