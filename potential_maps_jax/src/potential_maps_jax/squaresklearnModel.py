from squareData import make_squares_dataset

import jax.numpy as jnp

from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


RESOLUTION = 13
SIZE  = 36
NOISE_LO = -2
NOISE_HI = 2



trn, test = make_squares_dataset(n_train=64, n_test=132, noise_lo=NOISE_LO, noise_hi=NOISE_HI,
                 resolution=RESOLUTION, size=SIZE)


train_X = []
train_Y = []
test_X = []
test_Y = []
for x in trn:
    train_X.append(x[0].flatten())
    train_Y.append(x[1].flatten())
for z in test:
    test_X.append(z[0].flatten())
    test_Y.append(z[1].flatten())

base_train_X = jnp.stack(train_X)
base_train_Y = jnp.stack(train_Y)
base_test_X = jnp.stack(test_X)
base_test_Y = jnp.stack(test_Y)


regress1 = KNeighborsRegressor(n_neighbors=5)
regress1.fit(base_train_X, base_train_Y)

check1 = regress1.predict(base_test_X[131].reshape(1,-1)).reshape(2,-1).T
checx1 = base_test_X[131,:].reshape(2,-1).T
checy1 = base_test_Y[131,:].reshape(2,-1).T
fig ,ax = plt.subplots()
ax.set_aspect(1)
ax.scatter(checx1[:,0], checx1[:,1])
ax.scatter(checy1[:,0], checy1[:,1])
ax.scatter(check1[:,0], check1[:,1])
plt.show()
plt.close()

check2 = regress1.predict(base_test_X[67].reshape(1, -1)).reshape(2,-1).T
checx2 = base_test_X[67,:].reshape(2,-1).T
checy2 = base_test_Y[67,:].reshape(2,-1).T
fig ,ax = plt.subplots()
ax.set_aspect(1)
ax.scatter(checx2[:,0], checx2[:,1])
ax.scatter(checy2[:,0], checy2[:,1])
ax.scatter(check2[:,0], check2[:,1])
plt.show()
plt.close()
