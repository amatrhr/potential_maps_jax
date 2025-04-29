"""
Fitting  a neural net w/ FLAX NNX to "complete the square"
"""

from squareData import make_squares_dataset

from flax import nnx
import optax
import random
import jax.numpy as jnp
from sklearn.metrics import r2_score
learning_rate = 0.00625
momentum = 0.75
epochs = 512
batch_size = 256

RESOLUTION = 64
SIZE  = 21
NOISE_LO = -.3
NOISE_HI = .3

from IPython.display import clear_output
import matplotlib.pyplot as plt
eval_every = 32
metrics_history = {
    'train_loss': []
}

trn, test = make_squares_dataset(noise_lo=NOISE_LO, noise_hi=NOISE_HI,
                 resolution=RESOLUTION, size=SIZE)

base_train = [(x[0],x[1]) for x in trn]
base_test = [(x[0],x[1]) for x in test]

class LinearNN(nnx.Module):
    """A simple CNN model."""
    def __init__(self, *, rngs:nnx.Rngs):

        self.linear1 = nnx.Linear(2, 1025, rngs=rngs)
        # dropout
        self.linear2 = nnx.Linear(1025, 256, rngs=rngs)
        self.linear3 = nnx.Linear(256, 256, rngs=rngs)
        self.dropout = nnx.Dropout(0.95, rngs=rngs)
        self.linear3000 = nnx.Linear(256, 512, rngs=rngs)
        self.smalinear = nnx.Linear(512,2,rngs=rngs)
        # the in_feature dimension here was crucial to


    def __call__(self, x):
        # x = x.reshape(x.shape[0], -1) # flatten
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.linear3000(x)

        x = nnx.standardize(x)

        x = self.smalinear(x)
        x = SIZE* nnx.sigmoid(x) # this was a very important step for this problem.
        return x

# instantiate
model = LinearNN(rngs=nnx.Rngs(0))
# visualize it
nnx.display(model)

def loss_fn(model: LinearNN, batch):
    predictions = model(batch[0])
    loss = optax.losses.cosine_distance(
        predictions=predictions, targets=batch[1]
    ).mean()
    return loss, predictions

@nnx.jit
def train_step(model: LinearNN, optimizer: nnx.Optimizer, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, predictions), grads = grad_fn(model, batch)
    # metrics.update(loss=loss)
    optimizer.update(grads)

@nnx.jit
def eval_step(model: LinearNN, batch):
    loss, predictions = loss_fn(model, batch)
    # metrics.update(loss=loss)


optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
nnx.display(optimizer)



trn_loss = []
trn_r2 = []
test_loss = []
test_r2 = []

test_ds_X  = jnp.asarray([base_test[x][0] for x in range(len(base_test))])
test_ds_Y  = jnp.asarray([base_test[x][1] for x in range(len(base_test))])
test_ds = (test_ds_X, test_ds_Y)
# training loop
for epoch in range(epochs):
    trn_idx = random.choices(range(len(base_train)), k=batch_size)

    train_ds_X = jnp.asarray([base_train[x][0] for x in trn_idx])
    train_ds_Y = jnp.asarray([base_train[x][1] for x in trn_idx])
    train_ds = (train_ds_X, train_ds_Y)
    # need to make this into  a tuple of arrays

    train_step(model, optimizer, train_ds)
    trn_loss.append(loss_fn(model, train_ds)[0])
    # test_loss.append(loss_fn(model,test_ds)[0])
    temp_r2strn = []
    temp_r2stst = []
    for i in range(batch_size):
        temp_r2strn.append(r2_score((train_ds[0][i]), train_ds[1][i]))
    for j in range(len(base_test)):
        temp_r2stst.append(r2_score((test_ds[0][j]), test_ds[1][j]))
    trn_r2.append(jnp.nanmean(jnp.asarray(temp_r2strn)))
    test_r2.append(jnp.nanmean(jnp.asarray(temp_r2stst)))


# illustrate a thing

fig, ax = plt.subplots()
ax.plot(trn_loss, label='train')
ax.plot(test_loss, label='test')
ax.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(trn_r2, label='train')
ax.plot(test_r2, label='test')
ax.legend()
plt.show()
plt.close()

check1 = model(test_ds[0][131])

fig ,ax = plt.subplots(2,1)
ax[0].set_aspect(1)
ax[1].set_aspect(1)
ax[0].scatter(test_ds[0][131][:,0], test_ds[0][131][:,1], alpha=0.67, label = "test X")
ax[0].scatter(test_ds[1][131][:,0], test_ds[1][131][:,1], alpha=0.67, label = "test Y")
ax[0].scatter(check1[:,0], check1[:,1], alpha=0.67, label = "predicted Y")

ax[1].scatter(check1[:,0], check1[:,1],c="red", label="predicted Y")
fig.legend(loc='outside upper right')
plt.show()
plt.close()


check1 = model(test_ds[0][167])

fig ,ax = plt.subplots(2,1)
ax[0].set_aspect(1)
ax[1].set_aspect(1)
ax[0].scatter(test_ds[0][167][:,0], test_ds[0][167][:,1], alpha=0.67, label = "test X")
ax[0].scatter(test_ds[1][167][:,0], test_ds[1][167][:,1], alpha=0.67, label = "test Y")
ax[0].scatter(check1[:,0], check1[:,1], alpha=0.67, label = "predicted Y")

ax[1].scatter(check1[:,0], check1[:,1],c="red", label="predicted Y")
fig.legend(loc='outside upper right')
plt.show()
plt.close()


check1 = model(test_ds[0][41])

fig ,ax = plt.subplots(2,1)
ax[0].set_aspect(1)
ax[1].set_aspect(1)
ax[0].scatter(test_ds[0][41][:,0], test_ds[0][41][:,1], alpha=0.67, label = "test X")
ax[0].scatter(test_ds[1][41][:,0], test_ds[1][41][:,1], alpha=0.67, label = "test Y")
ax[0].scatter(check1[:,0], check1[:,1], alpha=0.67, label = "predicted Y")

ax[1].scatter(check1[:,0], check1[:,1],c="red", label="predicted Y")
fig.legend(loc='outside upper right')
plt.show()
plt.close()
