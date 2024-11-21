import numpy as np
import numpy as np
from openTSNE import affinity, TSNEEmbedding, initialization
from openTSNE.affinity import Affinities  # Import the class
from openTSNE.tsne import TSNE
from openTSNE import _tsne
from openTSNE.quad_tree import QuadTree

import matplotlib.gridspec as gridspec


import matplotlib.pyplot as plt
from dataclasses import dataclass
import seaborn as sns
import jax.numpy as jnp
from jax import grad
import jax


from cython_tsne_utils import quad_tree
from cython_tsne_utils import tsne_bh as custom_bh


@dataclass
class TSNEResult:
    """Dataclass to store the results of a single t-SNE algorithm run."""

    dataset_name: str
    n_samples: int
    n_iter: int
    embedding: np.ndarray
    optimization_mode: str | None
    kl_divergence: float
    initial_alpha: float
    alpha_lr: float | None
    im_embeddings: list
    im_KLs: np.ndarray
    im_alphas: np.ndarray
    im_alpha_grads: np.ndarray


def compute_low_dim_affinities_exact(
    Y: np.ndarray | TSNEEmbedding, α: float
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Computes the joint distribution of the embedded points Q (q_i_j) in the low-dimensional space.

    Parameters
    ----------
    Y: np.ndarray | TSNE.Embedding
        The low-dimensional representation of the data.
    alpha: float
        The degree-of-freedom parameter of the t-SNE algorithm.

    Notes
    -----
    The function is optimized for JAX for numerical computation of the gradients.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        - Q: The joint distribution of the embedded points in the low-dimensional space.
        - distances: The pairwise squared distances between all points in the low-dimensional space.
        - sum_Q: The sum of the joint distribution Q.

    """
    Y = jnp.asarray(Y)  # Convert to numpy array
    n = Y.shape[0]

    # Optimized squared distance computation
    sum_Y = jnp.sum(jnp.square(Y), 1)
    distances = -2.0 * jnp.dot(Y, Y.T)
    distances = jnp.add(jnp.add(distances, sum_Y).T, sum_Y)

    num = (1.0 + distances / α) ** -(α)  # not forget to divide by α
    num = num.at[jnp.diag_indices(n)].set(0.0)

    sum_Q = jnp.maximum(jnp.sum(num), 1e-12)
    # num[range(n), range(n)] = 0.0
    Q = num / jnp.sum(num)
    Q = jnp.maximum(Q, 1e-12)

    return Q, distances, sum_Q


def calculate_sum_Q_BH(
    embedding: TSNEEmbedding,
    α: float,
    reference_embedding: TSNEEmbedding = None,
    theta=0.5,
) -> float:
    """Computes the sum of the joint distribution Q in the low-dimensional space using the Barnes-Hut approximation of the openTSNE library.

    Parameters
    ----------
    embedding : TSNEEmbedding
        The low-dimensional representation of the data.
    alpha : float
        The degree-of-freedom parameter of the t-SNE algorithm.
    reference_embedding : TSNEEmbedding, optional
        The reference embedding (used only when we incorporate new points into the existing embedding), by default None
    theta : float, optional
        Parameter of the QuadTree algorithm, by default 0.5

    Returns
    -------
    float
        The sum of the joint distribution Q in the low-dimensional space.
    """

    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")
    tree = QuadTree(embedding)
    sum_Q_BH = _tsne.estimate_negative_gradient_bh(
        tree=tree,
        embedding=embedding,
        pairwise_normalization=reference_embedding,
        dof=α,
        theta=theta,
        gradient=gradient,
    )
    del tree
    return sum_Q_BH


def calculate_sum_Q_FFT(
    embedding: TSNEEmbedding,
    α: float,
    n_interpolation_points: int = 3,
    min_num_intervals: int = 50,
    ints_in_interval: int = 1,
) -> float:
    """Computes the sum of the joint distribution Q in the low-dimensional space using the Fast Fourier Transform approximation of the openTSNE library.

    Parameters
    ----------
    embedding : TSNEEmbedding
        The low-dimensional representation of the data.
    alpha : float
        The degree-of-freedom parameter of the t-SNE algorithm.
    n_interpolation_points : int, optional
        Parameter of the FFT algorithm, by default 3
    min_num_intervals : int, optional
        Parameter of the FFT algorithm, by default 50
    ints_in_interval : int, optional
        Parameter of the FFT algorithm, by default 1

    Returns
    -------
    float
        The sum of the joint distribution Q in the low-dimensional space.
    """
    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")
    sum_Q_FFT = _tsne.estimate_negative_gradient_fft_2d(
        embedding=embedding,
        gradient=gradient,
        dof=α,
        n_interpolation_points=n_interpolation_points,
        min_num_intervals=min_num_intervals,
        ints_in_interval=ints_in_interval,
    )
    return sum_Q_FFT


def KL_divergence_exact(
    alpha: float, p_ij: np.ndarray | Affinities, d_squared: np.ndarray
) -> float:
    """Computes the exact value of the t-SNE loss function (KL divergence) for a given embedding.

    Parameters
    ----------
    alpha : float
        The degree-of-freedom parameter of the t-SNE algorithm.
    p_ij : np.ndarray
        The input affinities matrix.
    d_squared : np.ndarray
        The pairwise squared distances between all points in the low-dimensional space (embedding).

    Returns
    -------
    float
        The value of the t-SNE loss function (KL divergence) for the given embedding.
    """

    n = p_ij.shape[0]
    num_Q = (1.0 + d_squared / alpha) ** (-alpha)
    num_Q = jnp.array(num_Q)
    num_Q = num_Q.at[jnp.diag_indices(n)].set(0.0)
    Q = num_Q / jnp.sum(num_Q)
    Q = jnp.maximum(Q, 1e-12)  # Avoid numerical instability

    kl = jnp.sum(p_ij * jnp.log(p_ij / Q))
    return kl


def compute_gradient_alpha_exact(
    p: Affinities | np.ndarray, d_squared: np.ndarray, α: float
) -> float:
    """Compute the exact gradient of the t-SNE loss function (KL) with respect to the degree-of-freedom parameter α.

    Parameters
    ----------
    p : Affinities | np.ndarray
        The input openTSNE affinities object or a numpy array containing the affinities matrix.
    d_squared : np.ndarray
        The pairwise squared distances between all points in the low-dimensional space.

    Returns
    -------
    gradient : float
        The gradient of the t-SNE loss function with respect to α.
    """

    # Ensure inputs are numpy arrays
    p = np.array(p)
    d_squared = np.array(d_squared)

    s = 1 + d_squared / α

    num = s ** (-α)  # numerator

    Z = np.sum(num)  # normalization factor

    q = num / Z

    log_s = np.log(s)

    frac = d_squared / (α + d_squared)

    diff = p - q

    gradient_components = diff * (log_s - frac)

    gradient = np.sum(gradient_components)

    return gradient


def gradient_alpha_jax(
    α: float, square_distances: np.ndarray, affinities: np.ndarray
) -> float:
    """Computes the gradient of the t-SNE loss function with respect to the degree-of-freedom parameter α using JAX.

    Parameters
    ----------
    square_distances : np.ndarray
        The pairwise squared distances between all points in the low-dimensional space.
    affinities : np.ndarray
        The affinities matrix.

    Returns
    -------
    float
        The gradient of the t-SNE loss function with respect to α.
    """

    alpha_grad_jax = grad(
        KL_divergence_exact,
        argnums=0,
    )
    grad_value = alpha_grad_jax(α, affinities, square_distances)

    return grad_value


def compute_gradient_alpha_bh(
    embedding: TSNEEmbedding,
    affinity_matrix: np.ndarray | Affinities,
    alpha: float,
    alpha_gradient: np.ndarray = np.zeros(1),
    theta: float = 0.5,
    num_threads: int = 1,
    pairwise_normalization: np.ndarray | None = None,
) -> float:
    """Computes the gradient of the t-SNE loss function with respect to the degree-of-freedom parameter α using the modified Barnes-Hut approach from the openTSNE library.

    Parameters
    ----------
    embedding : TSNEEmbedding
        The low-dimensional representation of the data.
    affinity_matrix : np.ndarray | Affinities
        The affinities matrix.
    alpha : float
        The degree-of-freedom parameter of the t-SNE algorithm.
    alpha_gradient : np.ndarray, optional
        One-element array, needed only to store and return the gradient value, by default np.zeros(1)
    theta : float, optional
        Parameter of the QuadTree algorithm, by default 0.5
    num_threads : int, optional
        The number of threads to use, by default -1
    pairwise_normalization : np.ndarray | None, optional
        Pairwise normalization, by default None

    Returns
    -------
    float
        The gradient of the t-SNE loss function with respect to α.
    """
    affinity_matrix = np.asarray(affinity_matrix)
    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")

    # alpha_gradient = np.zeros(1)

    tree = quad_tree.QuadTree(embedding)

    sum_Q_BH = custom_bh.estimate_negative_gradient_bh(
        tree=tree,
        embedding=embedding,
        gradient=gradient,
        p=affinity_matrix,
        theta=theta,
        dof=alpha,
        alpha_gradient=alpha_gradient,
        num_threads=num_threads,
        pairwise_normalization=pairwise_normalization,
    )

    del tree
    return alpha_gradient[0]


def tsne_with_dof_optimisation(
    X: np.ndarray,
    n_iter: int,
    initial_alpha: float,
    optimise_for_alpha: str | None = None,
    alpha_lr: float | None = None,
    dataset_name: str = "N/A",
    # grad_mode: str = "manual",
    verbose: bool = True,
    num_threads: int = 1,
) -> TSNEResult:
    """
    Parameters
    ----------
    X : np.ndarray
        The data to be embedded.
    n_iter : int
        The number of iterations to run the t-SNE algorithm.
    initial_alpha : float
        The initial degree-of-freedom parameter of the t-SNE algorithm.
    alpha_lr : float
        The learning rate for the optimization of the degree-of-freedom parameter.
    dataset_name : str
        The name of the dataset.
    optimise_for_alpha : str, optional
        The optimization method for the degree-of-freedom parameter. Options are 'exact', 'drift', and None. By default None.
    verbose : bool, optional
        Whether to print the progress of the algorithm. By default True.
    num_threads : int, optional
        The number of threads to use. By default 1.

    Returns
    -------
    TSNEResult
        A dataclass containing the results of the t-SNE algorithm
    """

    # store the intermediaries
    embeddings = []
    alphas = np.full(n_iter, float(initial_alpha))
    grad_alpha = 0.0
    alpha_grads = np.zeros(n_iter)
    KLs = np.zeros(n_iter)

    n_samples = X.shape[0]

    # step 1: initialize the tsne embedding
    print("Initializing the t-SNE embedding...")
    initial_embedding, affinities = initialize_tsne_embedding(
        X=X,
        initial_alpha=initial_alpha,
    )
    affinity_matrix = np.asarray(affinities)

    # step 2: run the early exaggeration phase
    embedding = run_early_exaggeration_phase(
        initial_embedding=initial_embedding,
        initial_alpha=initial_alpha,
    )

    # step 3: run the optimization
    print(f"Running optimization for {n_iter} iterations...")
    current_alpha = initial_alpha

    for i in range(n_iter):
        KLs[i] = embedding.kl_divergence

        if optimise_for_alpha:
            if not alpha_lr:
                raise ValueError(
                    "Please provide a learning rate for the optimization of the degree-of-freedom parameter."
                )
            if optimise_for_alpha != "bh":
                Q, d, sum_Q = compute_low_dim_affinities_exact(
                    Y=embedding, α=current_alpha
                )

            if optimise_for_alpha == "exact":
                # Compute the gradient w.r.t. alpha
                print("Computing the exact gradients...") if i == 0 else None
                grad_alpha = compute_gradient_alpha_exact(
                    p=affinity_matrix, d_squared=np.array(d), α=current_alpha
                )
                current_alpha -= alpha_lr * grad_alpha
                embedding.optimize(1, inplace=True, dof=current_alpha)

            elif optimise_for_alpha == "jax":
                print("Computing the gradients using JAX...") if i == 0 else None
                # Compute the gradient w.r.t. alpha using numerical differentiation with JAX
                grad_alpha = gradient_alpha_jax(
                    α=current_alpha, square_distances=d, affinities=affinity_matrix
                )
                current_alpha -= alpha_lr * grad_alpha
                embedding.optimize(1, inplace=True, dof=current_alpha)

            elif optimise_for_alpha == "bh":
                print("Optimizing with Barnes-Hut...") if i == 0 else None
                grad_alpha = compute_gradient_alpha_bh(
                    embedding=embedding,
                    affinity_matrix=affinity_matrix,
                    alpha=current_alpha,
                    num_threads=num_threads,
                )

                current_alpha -= alpha_lr * grad_alpha
                embedding.optimize(1, inplace=True, dof=current_alpha)

            elif optimise_for_alpha == "drift" and i > 0:
                print("Optimizing with drift...") if i == 0 else None
                if alpha_lr > 0.1:
                    print(
                        "Learning rate is too high for drift optimization. Please choose a smaller learning rate"
                    )

                im_embedding = embedding.optimize(
                    1, inplace=False, dof=current_alpha + alpha_lr
                )
                kl = im_embedding.kl_divergence
                if kl < KLs[i - 1]:
                    current_alpha += alpha_lr
                    grad_alpha = alpha_lr

                else:
                    current_alpha -= alpha_lr
                    grad_alpha = -alpha_lr
                embedding.optimize(1, inplace=True, dof=current_alpha)

        elif not optimise_for_alpha:
            print("Using fixed alpha...") if i == 0 else None
            # Optimize the embedding
            embedding.optimize(1, inplace=True, dof=current_alpha)

        # store the intermediaries
        KLs[i] = embedding.kl_divergence
        alpha_grads[i] = grad_alpha
        embeddings.append(embedding)
        alphas[i] = current_alpha

        if verbose:
            print(f"Iteration {i} out of {n_iter} done")
            print(f"Current KL divergence: {KLs[i]}")
            print(f"Current alpha: {current_alpha}")
            print(f"Current alpha gradient: {alpha_grads[i]}")
            print("-----------------------------------")

    return TSNEResult(
        n_iter=n_iter,
        embedding=embedding,
        kl_divergence=embedding.kl_divergence,
        initial_alpha=initial_alpha,
        alpha_lr=alpha_lr,
        im_embeddings=embeddings,
        im_KLs=KLs,
        im_alphas=alphas,
        im_alpha_grads=alpha_grads,
        dataset_name=dataset_name,
        n_samples=n_samples,
        optimization_mode=optimise_for_alpha,
    )


def run_reference_tsne(
    X: np.ndarray,
    n_iter: int,
    fixed_alpha: float,
    dataset_name,
) -> TSNEResult:
    """Runs the reference t-SNE algorithm from the openTSNE library on the input data.

    Parameters
    ----------
    X : np.ndarray
        The high-dimensional data to be embedded.

    Returns
    -------
    TSNEEmbedding
        The t-SNE embedding of the input data.
    """
    pca_init = initialization.pca(X=X, random_state=0)
    affinities = affinity.PerplexityBasedNN(data=X, perplexity=30, random_state=0)
    n_samples = X.shape[0]
    tsne = TSNE(
        n_components=2,
        n_iter=n_iter,
        random_state=0,
        n_jobs=-1,
        dof=fixed_alpha,
    )
    embedding = tsne.fit(X, initialization=pca_init, affinities=affinities)
    return TSNEResult(
        dataset_name=dataset_name,
        n_samples=n_samples,
        n_iter=n_iter,
        embedding=embedding,
        optimization_mode=None,
        kl_divergence=embedding.kl_divergence,
        initial_alpha=fixed_alpha,
        alpha_lr=None,
        im_embeddings=[],
        im_KLs=np.zeros(n_iter),
        im_alphas=np.ones(n_iter),
        im_alpha_grads=np.zeros(n_iter),
    )


def initialize_tsne_embedding(
    X: np.ndarray,
    initial_alpha: float,
    perplexity: int = 30,
    n_jobs: int = -1,
    random_state: int = 0,
    n_components: int = 2,
) -> tuple[TSNEEmbedding, np.ndarray]:
    """_summary_

    Parameters
    ----------
    X : np.ndarray
        High-dimensional data to be embedded.
    initial_alpha : float
        The initial degree-of-freedom parameter of the t-SNE algorithm.
    perplexity : int, optional
        The perplexity parameter of the t-SNE algorithm, by default 30
    n_jobs : int, optional
        The number of parallel jobs to run, by default -1
    random_state : int, optional
        The random state for reproducibility, by default 0
    n_components : int, optional
        The number of dimensions of the t-SNE embedding, by default 2

    Returns
    -------
    tuple[TSNEEmbedding, np.ndarray]
        - The initial t-SNE embedding.
        - The affinity matrix.
    """

    # step 1: compute the affinity matrix
    print("Computing the affinity matrix...")
    affinities_obj = affinity.PerplexityBasedNN(
        X,
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=n_jobs,
        random_state=random_state,
    )

    affinity_matrix = affinities_obj.P.toarray()
    affinity_matrix = np.maximum(affinity_matrix, 1e-12)

    # step 2: perfrom the pca initialization
    print("Performing the PCA initialization...")
    pca_init = initialization.pca(X, random_state=random_state)

    # step 3: initialize the tsne embedding
    print("Initializing the t-SNE embedding...")
    initial_embedding = TSNE(
        n_components=n_components, dof=initial_alpha
    ).prepare_initial(X=X, affinities=affinities_obj, initialization=pca_init)
    return initial_embedding, affinity_matrix


def run_early_exaggeration_phase(
    initial_embedding: TSNEEmbedding,
    initial_alpha: float,
    exagerration: int = 12,
    n_iter: int = 250,
) -> TSNEEmbedding:
    """Runs the standard early exaggeration phase of the t-SNE algorithm.

    Parameters
    ----------
    embedding : TSNEEmbedding
        The initial t-SNE embedding.
    initial_alpha : float
        The initial degree-of-freedom parameter of the t-SNE algorithm.
    exagerration : int, optional
        The exaggeration factor for the early exaggeration phase, by default 12
    n_iter : int, optional
        The number of iterations to run the early exagerration phase, by default 250

    Returns
    -------
    TSNEEmbedding
        The t-SNE embedding after the early exaggeration phase.
    """

    n_samples = initial_embedding.shape[0]

    default_learning_rate = n_samples / exagerration

    print(
        f"Performing the early exaggeration fase with exaggeration = {exagerration} and learning rate = {default_learning_rate} for {n_iter} iterations..."
    )
    embedding = initial_embedding.optimize(
        n_iter,
        exaggeration=exagerration,
        learning_rate=default_learning_rate,
        inplace=True,
        dof=initial_alpha,
    )
    return embedding


### PLOTTING FUNCTIONS ###


def plot_tsne_result(data: TSNEResult, labels: np.ndarray, additional_title: str = ""):
    fig = plt.figure(figsize=(10, 8))

    # set up axes
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.4])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    ax1.scatter(data.embedding[:, 0], data.embedding[:, 1], c=labels)
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.despine(bottom=True, left=True, ax=ax1)

    ax2.plot(data.im_KLs, color="blue")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("KL divergence", color="blue")
    sns.despine(bottom=False, left=False, ax=ax2)

    ax3.plot(data.im_alphas, color="red")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("alpha", color="red")
    sns.despine(bottom=False, left=False, ax=ax3)

    ax4.plot(data.im_alpha_grads, color="green", label="alpha gradient")
    ax4.hlines(0, 0, data.n_iter, color="red", linestyle="--", label="zero gradient")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("alpha gradient", color="green")
    sns.despine(bottom=False, left=False, ax=ax4)

    if data.optimization_mode:
        title_param_1 = "adaptive"
        title_param_2 = f"Initial alpha = {data.initial_alpha}. Alpha learning rate = {data.alpha_lr}"
        title_param_3 = f"Final alpha = {data.im_alphas[-1]:.2f}. Final Loss = {data.kl_divergence:.2f}"
    else:
        title_param_1 = "fixed"
        title_param_2 = (
            f"Alpha = {data.initial_alpha}. Final Loss = {data.kl_divergence:.2f}"
        )
        title_param_3 = ""

    plt.suptitle(
        f"t-SNE embedding with {title_param_1} degrees of freedom\n{data.dataset_name} with {data.n_samples} samples\n{title_param_2}\n{title_param_3}\n{additional_title}",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    plt.tight_layout()
    plt.show()


def plot_side_by_side(
    data1: TSNEResult,
    data2: TSNEResult,
    labels: np.ndarray,
    additional_title_1: str = "",
    additional_title_2: str = "",
):
    fig = plt.figure(figsize=(16, 8))

    # set up axes
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 0.4])

    # plot the first embedding

    ax1_1 = fig.add_subplot(gs[0, :3])
    ax1_2 = fig.add_subplot(gs[1, 0])
    ax1_3 = fig.add_subplot(gs[1, 1])
    ax1_4 = fig.add_subplot(gs[1, 2])

    ax1_1.scatter(data1.embedding[:, 0], data1.embedding[:, 1], c=labels)
    ax1_1.set_xticks([])
    ax1_1.set_yticks([])

    if data1.optimization_mode:
        title_param_1 = "adaptive"
        title_param_2 = f"Initial alpha = {data1.initial_alpha}. Alpha learning rate = {data1.alpha_lr}"
        title_param_3 = f"Final alpha = {data1.im_alphas[-1]:.2f}. Final Loss = {data1.kl_divergence:.2f}"
    else:
        title_param_1 = "fixed"
        title_param_2 = (
            f"Alpha = {data1.initial_alpha}. Final Loss = {data1.kl_divergence:.2f}"
        )
        title_param_3 = ""

    ax1_1.set_title(
        f"t-SNE embedding with {title_param_1} degrees of freedom\n{data1.dataset_name} with {data1.n_samples} samples\n{title_param_2}\n{title_param_3}\n{additional_title_1}",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    sns.despine(bottom=True, left=True, ax=ax1_1)

    ax1_2.plot(data1.im_KLs, color="blue")
    ax1_2.set_xlabel("Iteration")
    ax1_2.set_ylabel("KL divergence", color="blue")
    sns.despine(bottom=False, left=False, ax=ax1_2)

    ax1_3.plot(data1.im_alphas, color="red")
    ax1_3.set_xlabel("Iteration")
    ax1_3.set_ylabel("alpha", color="red")
    sns.despine(bottom=False, left=False, ax=ax1_3)

    ax1_4.plot(data1.im_alpha_grads, color="green", label="alpha gradient")
    ax1_4.hlines(0, 0, data1.n_iter, color="red", linestyle="--", label="zero gradient")
    ax1_4.set_xlabel("Iteration")
    ax1_4.set_ylabel("alpha gradient", color="green")
    sns.despine(bottom=False, left=False, ax=ax1_4)

    # plot the second embedding

    ax2_1 = fig.add_subplot(gs[0, 3:])
    ax2_2 = fig.add_subplot(gs[1, 3])
    ax2_3 = fig.add_subplot(gs[1, 4])
    ax2_4 = fig.add_subplot(gs[1, 5])

    ax2_1.scatter(data2.embedding[:, 0], data2.embedding[:, 1], c=labels)
    ax2_1.set_xticks([])
    ax2_1.set_yticks([])

    if data2.optimization_mode:
        title_param_1 = "adaptive"
        title_param_2 = f"Initial alpha = {data2.initial_alpha}. Alpha learning rate = {data2.alpha_lr}"
        title_param_3 = f"Final alpha = {data2.im_alphas[-1]:.2f}. Final Loss = {data2.kl_divergence:.2f}"
    else:
        title_param_1 = "fixed"
        title_param_2 = (
            f"Alpha = {data2.initial_alpha}. Final Loss = {data2.kl_divergence:.2f}"
        )
        title_param_3 = ""

    ax2_1.set_title(
        f"t-SNE embedding with {title_param_1} degrees of freedom\n{data2.dataset_name} with {data2.n_samples} samples\n{title_param_2}\n{title_param_3}\n{additional_title_2}",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    sns.despine(bottom=True, left=True, ax=ax2_1)

    ax2_2.plot(data2.im_KLs, color="blue")
    ax2_2.set_xlabel("Iteration")
    ax2_2.set_ylabel("KL divergence", color="blue")
    sns.despine(bottom=False, left=False, ax=ax2_2)

    ax2_3.plot(data2.im_alphas, color="red")
    ax2_3.set_xlabel("Iteration")
    ax2_3.set_ylabel("alpha", color="red")
    sns.despine(bottom=False, left=False, ax=ax2_3)

    ax2_4.plot(data2.im_alpha_grads, color="green", label="alpha gradient")
    ax2_4.hlines(0, 0, data2.n_iter, color="red", linestyle="--", label="zero gradient")
    ax2_4.set_xlabel("Iteration")
    ax2_4.set_ylabel("alpha gradient", color="green")
    sns.despine(bottom=False, left=False, ax=ax2_4)

    plt.tight_layout()

    plt.show()
