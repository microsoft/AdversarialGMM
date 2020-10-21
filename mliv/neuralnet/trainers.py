# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sklearn.cluster import KMeans
from .agmm_earlystop import AGMMEarlyStop, KernelLossAGMMEarlyStop, CentroidMMDGMMEarlyStop, KernelLayerMMDGMMEarlyStop
from .architectures import CNN_Z_agmm, CNN_Z_kernel, CNN_X, CNN_X_bn, fc_z_kernel, fc_z_agmm, fc_x
from .utilities import log_metrics, dprint
from .rbflayer import gaussian, inverse_multiquadric

# train AGMM


def train_agmm(
    Z_train,
    T_train,
    Y_train,
    G_train,
    Z_dev,
    T_dev,
    Y_dev,
    G_dev,
    Z_val,
    T_val,
    Y_val,
    G_val,
    T_test,
    G_test,
    X_IMAGE=False,
    Z_IMAGE=False,
    n_t=1,
    n_instruments=2,
    n_hidden=200,
    dropout_p=0.1,
    learner_lr=1e-4,
    adversary_lr=1e-4,
    learner_l2=1e-4,
    adversary_l2=1e-4,
    adversary_norm_reg=1e-4,
    n_epochs=100,
    batch_size=100,
    train_learner_every=1,
    train_adversary_every=1,
    device=None,
    DEBUG=False,
):
    if X_IMAGE:
        learner = CNN_X()
    else:
        learner = fc_x(n_t, n_hidden, dropout_p)
    if Z_IMAGE:
        adversary = CNN_Z_agmm()
    else:
        adversary = fc_z_agmm(n_instruments, n_hidden, dropout_p)

    def logger(learner, adversary, epoch, writer):
        if not X_IMAGE:
            writer.add_histogram("learner", learner[-1].weight, epoch)
        if not Z_IMAGE:
            writer.add_histogram("adversary", adversary[-1].weight, epoch)
        log_metrics(
            Z_val,
            T_val,
            Y_val,
            Z_val,
            T_val,
            Y_val,
            T_test,
            learner,
            adversary,
            epoch,
            writer,
            true_of_T=G_val,
        )

    # np.random.seed(12356)
    dprint(DEBUG, "---Hyperparameters---")
    dprint(DEBUG, "Learner Learning Rate:", learner_lr)
    dprint(DEBUG, "Adversary learning rate:", adversary_lr)
    dprint(DEBUG, "Learner_l2:", learner_l2)
    dprint(DEBUG, "Adversary_l2:", adversary_l2)
    dprint(DEBUG, "Number of epochs:", n_epochs)
    dprint(DEBUG, "Batch Size:", batch_size)
    agmm = AGMMEarlyStop(learner, adversary).fit(
        Z_train,
        T_train,
        Y_train,
        Z_dev,
        T_dev,
        Y_dev,
        learner_lr=learner_lr,
        adversary_lr=adversary_lr,
        learner_l2=learner_l2,
        adversary_l2=adversary_l2,
        n_epochs=n_epochs,
        bs=batch_size,
        logger=logger,
        model_dir="agmm_model",
        device=device,
        train_learner_every=train_learner_every,
        train_adversary_every=train_adversary_every,
    )

    return agmm


# Train KernelLayerGMM
def train_kernellayergmm(
    Z_train,
    T_train,
    Y_train,
    G_train,
    Z_dev,
    T_dev,
    Y_dev,
    G_dev,
    Z_val,
    T_val,
    Y_val,
    G_val,
    T_test,
    G_test,
    g_features=100,
    n_centers=100,
    kernel_fn=gaussian,
    centers=None,
    sigmas=None,
    X_IMAGE=False,
    Z_IMAGE=False,
    n_t=1,
    n_instruments=2,
    n_hidden=200,
    dropout_p=0.1,
    learner_lr=1e-4,
    adversary_lr=1e-4,
    learner_l2=1e-4,
    adversary_l2=1e-4,
    adversary_norm_reg=1e-4,
    n_epochs=100,
    batch_size=100,
    train_learner_every=1,
    train_adversary_every=1,
    device=None,
    DEBUG=False,
):
    if X_IMAGE:
        learner = CNN_X()
    else:
        learner = fc_x(n_t, n_hidden, dropout_p)
    if Z_IMAGE:
        adversary = CNN_Z_kernel(g_features)
    else:
        adversary = fc_z_kernel(n_instruments, n_hidden, g_features, dropout_p)

    def logger(learner, adversary, epoch, writer):
        if not X_IMAGE:
            writer.add_histogram("learner", learner[-1].weight, epoch)
        # if not Z_IMAGE:
        #  writer.add_histogram('adversary', adversary[-1].weight, epoch)
        writer.add_histogram("adversary", adversary.beta.weight, epoch)
        log_metrics(
            Z_val,
            T_val,
            Y_val,
            Z_val,
            T_val,
            Y_val,
            T_test,
            learner,
            adversary,
            epoch,
            writer,
            true_of_T=G_val,
        )

    # np.random.seed(12356)
    dprint(DEBUG, "---Hyperparameters---")
    dprint(DEBUG, "Learner Learning Rate:", learner_lr)
    dprint(DEBUG, "Adversary learning rate:", adversary_lr)
    dprint(DEBUG, "Learner_l2:", learner_l2)
    dprint(DEBUG, "Adversary_l2:", adversary_l2)
    dprint(DEBUG, "Number of epochs:", n_epochs)
    dprint(DEBUG, "Batch Size:", batch_size)
    dprint(DEBUG, "G features", g_features)
    dprint(DEBUG, "Kernel function", kernel_fn.__name__)
    klayermmdgmm = KernelLayerMMDGMMEarlyStop(
        learner,
        adversary,
        g_features,
        n_centers,
        kernel_fn,
        centers=centers,
        sigmas=sigmas,
    )
    klayermmdgmm.fit(
        Z_train,
        T_train,
        Y_train,
        Z_dev,
        T_dev,
        Y_dev,
        learner_l2=learner_l2,
        adversary_l2=adversary_l2,
        adversary_norm_reg=adversary_norm_reg,
        learner_lr=learner_lr,
        adversary_lr=adversary_lr,
        n_epochs=n_epochs,
        bs=batch_size,
        logger=logger,
        model_dir="klayer_model",
        device=device,
        train_learner_every=train_learner_every,
        train_adversary_every=train_adversary_every,
    )

    return klayermmdgmm


# Train CentroidMMDGMM


def train_centroidmmdgmm(
    Z_train,
    T_train,
    Y_train,
    G_train,
    Z_dev,
    T_dev,
    Y_dev,
    G_dev,
    Z_val,
    T_val,
    Y_val,
    G_val,
    T_test,
    G_test,
    n_centers=100,
    kernel_fn=gaussian,
    sigma=None,
    X_IMAGE=False,
    Z_IMAGE=False,
    n_t=1,
    n_instruments=2,
    n_hidden=200,
    dropout_p=0.1,
    learner_lr=1e-4,
    adversary_lr=1e-4,
    learner_l2=1e-4,
    adversary_l2=1e-4,
    adversary_norm_reg=1e-4,
    n_epochs=100,
    batch_size=100,
    train_learner_every=1,
    train_adversary_every=1,
    device=None,
    DEBUG=False,
):
    if X_IMAGE:
        learner = CNN_X()
    else:
        learner = fc_x(n_t, n_hidden, dropout_p)
    if Z_IMAGE:
        adversary = CNN_Z_kernel(g_features)
    else:
        adversary = fc_z_kernel(n_instruments, n_hidden, g_features, dropout_p)

    centers = KMeans(n_clusters=n_centers).fit(
        Z_train.reshape(Z_train.shape[0], -1)).cluster_centers_
    centers = centers.reshape([n_centers]+list(Z_train.shape[1:]))

    def logger(learner, adversary, epoch, writer):
        # if not X_IMAGE:
            #writer.add_histogram("learner", learner[-1].weight, epoch)
        # if not Z_IMAGE:
        #  writer.add_histogram('adversary', adversary[-1].weight, epoch)
        #writer.add_histogram("adversary", adversary.beta.weight, epoch)
        log_metrics(
            Z_val,
            T_val,
            Y_val,
            Z_val,
            T_val,
            Y_val,
            T_test,
            learner,
            adversary,
            epoch,
            writer,
            true_of_T=G_val,
        )

    # np.random.seed(12356)
    dprint(DEBUG, "---Hyperparameters---")
    dprint(DEBUG, "Learner Learning Rate:", learner_lr)
    dprint(DEBUG, "Adversary learning rate:", adversary_lr)
    dprint(DEBUG, "Learner_l2:", learner_l2)
    dprint(DEBUG, "Adversary_l2:", adversary_l2)
    dprint(DEBUG, "Number of epochs:", n_epochs)
    dprint(DEBUG, "Batch Size:", batch_size)
    dprint(DEBUG, "Number of centers", n_centers)
    dprint(DEBUG, "Kernel function", kernel_fn.__name__)
    centroidmmdgmm = CentroidMMDGMMEarlyStop(
        learner,
        adversary,
        kernel_fn,
        centers,
        np.ones(n_centers)*sigma,
    )
    centroidmmdgmm.fit(
        Z_train,
        T_train,
        Y_train,
        Z_dev,
        T_dev,
        Y_dev,
        learner_l2=learner_l2,
        adversary_l2=adversary_l2,
        adversary_norm_reg=adversary_norm_reg,
        learner_lr=learner_lr,
        adversary_lr=adversary_lr,
        n_epochs=n_epochs,
        bs=batch_size,
        logger=logger,
        model_dir="centroid_model",
        device=device,
        train_learner_every=train_learner_every,
        train_adversary_every=train_adversary_every,
    )

    return centroidmmdgmm


# Train KernelLossAGMM
def train_kernellossagmm(
    Z_train,
    T_train,
    Y_train,
    G_train,
    Z_dev,
    T_dev,
    Y_dev,
    G_dev,
    Z_val,
    T_val,
    Y_val,
    G_val,
    T_test,
    G_test,
    kernel_fn=gaussian,
    sigma=None,
    X_IMAGE=False,
    Z_IMAGE=False,
    n_t=1,
    n_instruments=2,
    n_hidden=200,
    dropout_p=0.1,
    learner_lr=1e-4,
    adversary_lr=1e-4,
    learner_l2=1e-4,
    adversary_l2=1e-4,
    n_epochs=100,
    batch_size=100,
    train_learner_every=1,
    train_adversary_every=1,
    device=None,
    DEBUG=False,
):
    if X_IMAGE:
        learner = CNN_X()
    else:
        learner = fc_x(n_t, n_hidden, dropout_p)
    if Z_IMAGE:
        adversary = CNN_Z_kernel(g_features)
    else:
        adversary = fc_z_kernel(n_instruments, n_hidden, g_features, dropout_p)

    def logger(learner, adversary, epoch, writer):
        #writer.add_histogram('learner', learner[-1].weight, epoch)
        #writer.add_histogram('adversary', adversary.sigma, epoch)
        log_metrics(Z_val, T_val, Y_val, Z_val, T_val, Y_val, T_test,
                    learner, adversary, epoch, writer, true_of_T=G_val, loss='kernel')

    # np.random.seed(12356)
    dprint(DEBUG, "---Hyperparameters---")
    dprint(DEBUG, "Learner Learning Rate:", learner_lr)
    dprint(DEBUG, "Adversary learning rate:", adversary_lr)
    dprint(DEBUG, "Learner_l2:", learner_l2)
    dprint(DEBUG, "Adversary_l2:", adversary_l2)
    dprint(DEBUG, "Number of epochs:", n_epochs)
    dprint(DEBUG, "Batch Size:", batch_size)
    dprint(DEBUG, "Kernel function", kernel_fn.__name__)
    dprint(DEBUG, "Sigma", sigma)

    kernellossagmm = KernelLossAGMMEarlyStop(
        learner, adversary, kernel_fn, sigma)
    kernellossagmm.fit(Z_train, T_train, Y_train, Z_dev, T_dev, Y_dev,
                       learner_l2=learner_l2**2, adversary_l2=adversary_l2,
                       learner_lr=learner_lr, adversary_lr=adversary_lr, n_epochs=n_epochs,
                       bs=batch_size, logger=logger, model_dir='kernel_model', device=device)
    return kernellossagmm
