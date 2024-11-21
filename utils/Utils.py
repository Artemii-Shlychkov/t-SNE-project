import numpy as np
from typing import Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import sklearn.datasets
from typing import Tuple, List
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import openTSNE as TSNE
from openTSNE.initialization import pca as pca
from typing import List
from dataclasses import dataclass, asdict
from openTSNE.affinity import PerplexityBasedNN
from openTSNE.nearest_neighbors import NNDescent
import pandas as pd
import json
import pickle
# from sklearn.decomposition import PCA


@dataclass
class TSNEResult:
    raw_data: np.ndarray
    x: np.ndarray
    y: np.ndarray
    alpha: float
    n_samples = None
    labels: np.ndarray | None
    KL: float | None
    kNN_recall: float | None

    def __post_init__(self):
        self.n_samples = self.raw_data.shape[0]


@dataclass
class DataSetTSNE:
    dataset_name: str
    tsne_results: List[TSNEResult]
    alphas: List[float] = None
    KLs: List[float] = None
    min_KL: float = 0
    optimal_alpha_KL: float = 0
    max_kNN_recall: float = 0
    kNNs: List[float] = None
    optimal_alpha_kNN: float = 0

    def __post_init__(self):
        self.alphas = [alpha for alpha in self.alphas]
        self.KLs = [result.KL for result in self.tsne_results]
        self.min_KL = min([result.KL for result in self.tsne_results])
        self.optimal_alpha_KL = self.alphas[
            np.argmin([result.KL for result in self.tsne_results])
        ]
        self.kNNs = [result.kNN_recall for result in self.tsne_results]
        self.max_kNN_recall = max([result.kNN_recall for result in self.tsne_results])
        self.optimal_alpha_kNN = self.alphas[
            np.argmax([result.kNN_recall for result in self.tsne_results])
        ]


@dataclass
class GoodnessOfFit:
    alphas: np.ndarray
    KL_divergence: list
    optimal_KL = None
    optimal_alpha_KL = None
    kNN_recall: list
    optimal_kNN_recall = None
    optimal_alpha_kNN_recall = None

    def __post_init__(self):
        self.optimal_KL = min(self.KL_divergence)
        self.optimal_alpha_KL = self.alphas[self.KL_divergence.index(self.optimal_KL)]
        self.optimal_kNN_recall = max(self.kNN_recall)
        self.optimal_alpha_kNN_recall = self.alphas[
            self.kNN_recall.index(self.optimal_kNN_recall)
        ]


def compute_goodness_of_fit(
    raw_data: np.ndarray,
    tsne_data_list: List[TSNE.TSNE],
    dofs: np.ndarray,
    plot: bool = False,
) -> GoodnessOfFit:
    """
    Computes the optimal degrees of freedom based on the specified metric.

    Parameters
    ----------
    raw_data : np.ndarray
        _description_
    tsne_data_list : List[TSNE.TSNE]
        _description_
    dofs : np.ndarray
        _description_
    metric : str
        _description_
    plot : bool, optional
        _description_, by default False

    Returns
    -------
    Tuple[int, float]
        _description_
    """

    assert len(tsne_data_list) == len(
        dofs
    ), "Length of tsne_data_list and dofs must be the same"

    kNN_recall = [
        compute_knn_recall(raw_data, tsne_data) for tsne_data in tsne_data_list
    ]
    KL_divergence = [tsne_data.kl_divergence for tsne_data in tsne_data_list]

    results = GoodnessOfFit(
        KL_divergence=KL_divergence, kNN_recall=kNN_recall, alphas=dofs
    )

    if plot:
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # ax[0].plot(dofs, KL_divergence, label="KL")
        # ax[0].set_xlabel("Degrees of Freedom")
        # ax[0].set_ylabel("KL Divergence")
        # ax[0].legend()
        # ax[0].scatter(
        #     results.optimal_alpha_KL,
        #     results.optimal_KL,
        #     color="red",
        #     label="Optimal",
        # )
        # # use log scale for KL divergence
        # ax[0].set_xscale("log")

        # ax[1].plot(dofs, kNN_recall, label="kNN Recall")
        # ax[1].set_xlabel("Degrees of Freedom")
        # ax[1].set_ylabel("kNN Recall")
        # ax[1].legend()
        # ax[1].scatter(
        #     results.optimal_alpha_kNN_recall,
        #     results.optimal_kNN_recall,
        #     color="red",
        #     label="Optimal",
        # )
        # ax[1].set_xscale("log")

        # sns.despine()
        # plt.show()

        # sns.despine()
        # plt.show()

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("KL Divergence", "kNN Recall"),
            horizontal_spacing=0.1,
        )
        fig.add_trace(
            go.Scatter(x=dofs, y=KL_divergence, mode="lines", name="KL Divergence"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=dofs, y=kNN_recall, mode="lines", name="kNN Recall"),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[results.optimal_alpha_KL],
                y=[results.optimal_KL],
                mode="markers+text",
                name="Optimal KL",
                marker=dict(color="white"),
                text=[f"α={results.optimal_alpha_KL:.2f}"],
                textfont=dict(color="white", size=12),
                textposition="bottom center",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[results.optimal_alpha_kNN_recall],
                y=[results.optimal_kNN_recall],
                mode="markers+text",
                name="Optimal kNN Recall",
                marker=dict(color="white"),
                text=[f"α={results.optimal_alpha_kNN_recall:.2f}"],
                textfont=dict(color="white", size=12),
                textposition="top center",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Degrees of Freedom", row=1, col=1)
        fig.update_xaxes(title_text="Degrees of Freedom", row=1, col=2)
        fig.update_yaxes(title_text="Metric value", row=1, col=1)

        for i in range(1, 3):
            fig.update_xaxes(
                type="log",
                row=1,
                col=i,
                tickfont=dict(
                    family="Arial", size=15, color="white"
                ),  # Major ticks font
                tickvals=[1, 10, 100],  # Powers of 10
                ticktext=["10⁰", "10¹", "10²"],
            )

        fig.update_layout(
            title="Goodness of t-SNE fit for the dataset with different DoFs",
            showlegend=False,
            template="plotly_dark",
            width=1000,
            height=400,
            font=dict(
                family="Courier New, monospace",
                size=15,
            ),
        )
        fig.show()

    return results


def compute_knn_recall(
    original_data: np.ndarray, tsne_data: np.ndarray, k: int = 10
) -> float:
    """
    Computes the recall of k-nearest neighbors between the original data and the t-SNE data.

    Parameters
    ----------
    original_data : np.ndarray
        The original multidimensional data.
    tsne_data : np.ndarray
        The t-SNE transformed data.
    k : int, optional
        The number of neighbors to consider, by default 7

    Returns
    -------
    float
        The average recall of k-nearest neighbors between the original data and the t-SNE data.

    Notes
    -----

    The formula is taken from: Gove et al. (2022)
    New guidance for using t-SNE: Alternative defaults, hyperparameter selection automation,
    and comparative evaluation,
    Visual Informatics, Volume 6, Issue 2, 2022,

    """
    # Fit kNN on original data
    knn_orig = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_orig.fit(original_data)
    orig_neighbors = knn_orig.kneighbors(return_distance=False)

    # Fit kNN on t-SNE data
    knn_tsne = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_tsne.fit(tsne_data)
    tsne_neighbors = knn_tsne.kneighbors(return_distance=False)

    # Calculate recall for each point
    recall_scores = np.zeros(len(original_data))
    for i in range(len(original_data)):
        shared_neighbors = np.intersect1d(orig_neighbors[i], tsne_neighbors[i])
        recall = len(shared_neighbors) / k
        recall_scores[i] = recall
    # Return average recall
    return np.mean(recall_scores)


def generate_swiss_roll(
    n_samples: int = 1000, noise: float = 0.0, plot: bool = str, save_fig: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    n_samples : int, optional
        number of samples to generate, by default 1000
    noise : float, optional
        add noise to the swiss roll, by default 0.0
    plot : bool, optional
        whether to plot the swiss roll, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        coordinates of the swiss roll and the color of the points
    """

    sr_points, sr_color = sklearn.datasets.make_swiss_roll(
        n_samples=n_samples, noise=noise, random_state=42
    )
    if plot == "matplotlib":
        fig = plot_swiss_roll_matplotlib(
            sr_points=sr_points, sr_color=sr_color, n_samples=n_samples
        )
        fig.show()
        if save_fig:
            fig.savefig("figures/swiss_roll_matplotlib.png")
    elif plot == "plotly":
        fig = plot_swiss_roll_plotly(
            sr_points=sr_points, sr_color=sr_color, n_samples=n_samples
        )
        fig.show()
        if save_fig:
            fig.write_image("figures/swiss_roll_plotly.png")

    return sr_points, sr_color


def plot_swiss_roll_matplotlib(
    sr_points: np.ndarray, sr_color: np.ndarray, n_samples: int
) -> plt.Figure:
    """
    Creates a 3D scatter plot of the Swiss Roll dataset using Matplotlib.

    Parameters
    ----------
    sr_points : np.ndarray
        The Swiss Roll dataset
    sr_color : np.ndarray
        The color of the points in the Swiss Roll dataset
    n_samples : int
        The number of samples in the Swiss Roll dataset

    Returns
    -------
    plt.Figure
        The Matplotlib figure object
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Set the face color of the figure and axes to dark
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.scatter(
        sr_points[:, 0],
        sr_points[:, 1],
        sr_points[:, 2],
        c=sr_color,
        cmap="rainbow",
        s=8,
    )

    ax.set_title("Swiss Roll in Ambient Space", color="white")
    ax.view_init(azim=-66, elev=12)
    _ = ax.text2D(
        0.8, 0.05, s=f"N samples={n_samples}", transform=ax.transAxes, color="white"
    )

    # Set the tick and label colors to light blue
    ax.tick_params(colors="lightblue", labelcolor="white", width=0.25)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")

    ax.grid(True)
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))

    # Set the grid color to light blue
    ax.xaxis._axinfo["grid"].update(
        {
            "color": "lightblue",
            "linewidth": 0.25,
        }
    )
    ax.yaxis._axinfo["grid"].update(
        {
            "color": "lightblue",
            "linewidth": 0.25,
        }
    )
    ax.zaxis._axinfo["grid"].update(
        {
            "color": "lightblue",
            "linewidth": 0.25,
        }
    )

    plt.show()
    return fig


# Create a 3D scatter plot
def plot_swiss_roll_plotly(
    sr_points: np.ndarray,
    sr_color: np.ndarray,
    n_samples: int,
    row=1,
    col=1,
    fig=None,
    title=None,
    width=800,
    height=600,
) -> go.Figure:
    """
    Create a 3D scatter plot of the Swiss Roll dataset using Plotly.

    Parameters
    ----------
    sr_points : np.ndarray
        The Swiss Roll dataset points.
    sr_color : np.ndarray
        The colors corresponding to each point.
    n_samples : int
        The number of samples in the dataset.
    row : int, optional
        The row position in a subplot grid. Default is 1.
    col : int, optional
        The column position in a subplot grid. Default is 1.
    fig : go.Figure, optional
        An existing figure object to add the Swiss Roll plot into.

    Returns
    -------
    go.Figure
        The Plotly figure object with the Swiss Roll 3D plot.
    """

    # If no existing figure is passed, create a new one
    if fig is None:
        fig = make_subplots(rows=row, cols=col, specs=[[{"type": "scene"}] * col] * row)

    # Add the Swiss Roll 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=sr_points[:, 0],
            y=sr_points[:, 1],
            z=sr_points[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=sr_color,
                colorscale="Rainbow",  # Choose a colorscale
                opacity=0.8,
            ),
        ),
        row=row,
        col=col,
    )
    title_font_size = width // 75
    # Update the layout for a dark background
    fig.update_layout(
        title="Swiss Roll in Ambient Space" if title is None else title,
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            camera=dict(
                eye=dict(x=-1, y=2, z=0.5),  # Camera perspective
            ),
        ),
        width=width * col,
        height=height * row,
        font=dict(family="Courier New, monospace", size=title_font_size),
    )

    # Add a text annotation for the number of samples
    fig.add_annotation(
        text=f"N samples={n_samples}",
        xref="paper",
        yref="paper",
        x=1,
        y=0.05,
        showarrow=False,
        font=dict(color="white"),
    )
    fig.show()
    return fig


def plot_TSNE(
    tsne_results: List[TSNEResult],
    display_metrics: bool = False,
    custom_alphas: Optional[np.ndarray] = None,
    grid=None,
    title=None,
    width=1600,
    height=800,
    show: bool = True,
    use_custom_colors: bool = False,
    colorscale="Rainbow",
    labeled: bool = True,
) -> go.Figure:
    if custom_alphas:
        alphas = custom_alphas
        # return only tsne_results for chosen alphas
        tsne_results = [tsne for tsne in tsne_results if tsne.alpha in custom_alphas]

    else:
        alphas = np.array([tsne.alpha for tsne in tsne_results])

    n_subplots = len(alphas)
    n_samples = tsne_results[0].n_samples

    # Grid auto-calculation if not provided
    if grid is None:
        n_cols = max((len(tsne_results) // 2), 1)
        n_rows = (n_subplots + n_cols - 1) // n_cols
    else:
        n_rows, n_cols = grid

    # Set color options

    custom_colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "lime",
        "white",
    ]

    # Prepare subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"α={alpha:.2f}"
            for alpha in (alphas if alphas is not None else range(n_subplots))
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.05,
    )

    # Add t-SNE scatter plots

    for i, tsne_result in enumerate(tsne_results):
        labels = tsne_result.labels
        if labeled and labels is not None:
            unique_labels = np.unique(labels)
            for k, label in enumerate(unique_labels):
                fig.add_trace(
                    go.Scatter(
                        x=tsne_result.x[labels == label],
                        y=tsne_result.y[labels == label],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=custom_colors[k % len(custom_colors)]
                            if use_custom_colors
                            else label,
                            opacity=0.8,
                        ),
                        name=f"Label {label}",
                        showlegend=True if i == 0 else False,
                    ),
                    row=(i // n_cols) + 1,
                    col=(i % n_cols) + 1,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=tsne_result.x,
                    y=tsne_result.y,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=labels if labels is not None else "gray",
                        colorscale=colorscale,
                        opacity=0.8,
                    ),
                    showlegend=False,
                ),
                row=(i // n_cols) + 1,
                col=(i % n_cols) + 1,
            )

        # Add metrics as annotations if required
        if display_metrics:
            xref = f"x{i+1}" if i > 0 else "x"
            yref = f"y{i+1}" if i > 0 else "y"
            fig.add_annotation(
                text=f"KL: {tsne_result.KL:.2f}",
                xref=f"{xref} domain",
                yref=f"{yref} domain",
                x=0,
                y=-0.1,
                showarrow=False,
                font=dict(color="white", size=12),
            )
            fig.add_annotation(
                text=f"kNN Recall: {tsne_result.kNN_recall:.2f}",
                xref=f"{xref} domain",
                yref=f"{yref} domain",
                x=0,
                y=-0.2,
                showarrow=False,
                font=dict(color="white", size=12),
            )

    # Set up title and layout
    sup_title = (
        f"t-SNE embedding of the dataset ({n_samples} datapoints)"
        if title is None
        else title
    )

    # dynamic font size
    title_font_size = width // 85

    fig.update_layout(
        title=sup_title,
        template="plotly_dark",
        width=width,
        height=height,
        font=dict(family="Courier New, monospace", size=title_font_size),
        margin=dict(l=0, r=0, t=100, b=100),
    )

    # Hide ticks and grids
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    if show:
        fig.show()

    return fig


def compute_tsne_embedding(
    raw_data: np.ndarray,
    alphas: np.ndarray,
    labels: Optional[np.ndarray] = None,
    perplexity: int = 50,
    dataset_name: Optional[str] = None,
    compute_metrics: bool = False,
    save_results: bool = False,
) -> List[TSNEResult]:
    print("Computing the shared affinities...")

    raw_data_name = str()

    affinities = PerplexityBasedNN(
        raw_data,
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=-1,
    )

    print("Computing the PCA initialization...")
    pca_init = pca(X=raw_data, n_components=2, random_state=42, svd_solver="auto")

    tsne_results = []

    for alpha in alphas:
        print(f"Computing t-SNE embedding for alpha={alpha}...")
        tsne = TSNE.TSNE(
            perplexity=perplexity,
            initialization=pca_init,
            n_jobs=-1,
            random_state=42,
            dof=alpha,
            metric="euclidean",
        ).fit(X=None, affinities=affinities)

        KL = None
        kNN_recall = None

        if compute_metrics:
            KL = tsne.kl_divergence
            kNN_recall = compute_knn_recall(raw_data, tsne)

        x_coords = np.array(tsne[:, 0])
        y_coords = np.array(tsne[:, 1])

        embedding_result = TSNEResult(
            raw_data=raw_data,
            x=x_coords,
            y=y_coords,
            labels=labels,
            alpha=alpha,
            KL=KL,
            kNN_recall=kNN_recall,
        )

        tsne_results.append(embedding_result)

    if save_results:
        dataset_name = dataset_name if dataset_name is not None else "dataset"
        print("Saving results to a .pkl file...")
        # Save results to a .npz file
        pickle.dump(tsne_results, open(f"{dataset_name}_tsne_results.pkl", "wb"))

    return tsne_results
