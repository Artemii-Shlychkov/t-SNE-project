# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport sqrt, log
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from quad_tree cimport QuadTree, Node, is_close

cdef double EPSILON = 1e-12

cpdef double estimate_negative_gradient_bh(
    QuadTree tree,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double[:, ::1] p,
    double theta,
    double dof,
    double[:] alpha_gradient,
    Py_ssize_t num_threads,
    bint pairwise_normalization
):
    """
    Estimate the negative gradient using Barnes-Hut approximation.
    """
    cdef Py_ssize_t i, j, num_points, n, d
    cdef double sum_Q = 0.0

    num_points = embedding.shape[0]

    # Memoryviews for distances and qij
    cdef double[:, :] distances = np.zeros((num_points, num_points), dtype=np.float64)
    cdef double[:, :] qij = np.zeros((num_points, num_points), dtype=np.float64)

    # Zero-initialize the gradient
    gradient[:, :] = 0.0

    # Compute gradients in parallel
    for i in prange(num_points, num_threads=num_threads, nogil=True):
        _estimate_negative_gradient_single(
            &tree.root,
            &embedding[i, 0],
            &gradient[i, 0],
            &sum_Q,
            theta,
            dof,
            distances,
            qij,
            i
        )

    # Normalize the gradient
    n = gradient.shape[0]
    d = gradient.shape[1]
    if pairwise_normalization:
        for i in range(n):
            for j in range(d):
                gradient[i, j] = gradient[i, j] / (sum_Q + EPSILON)

    # Compute alpha gradient if provided
    if alpha_gradient is not None:
        alpha_gradient[0] = compute_alpha_gradient(p, distances, qij, num_points, dof, sum_Q)

    return sum_Q



cdef void _estimate_negative_gradient_single(
    Node *node,
    double *point,
    double *gradient,
    double *sum_Q,
    double theta,
    double dof,
    double[:, :] distances,
    double[:, :] qij,
    Py_ssize_t i
) noexcept nogil:
    """
    Recursive function to estimate the negative gradient for a single point.
    """
    if node.num_points == 0 or (node.is_leaf and is_close(node, point, EPSILON)):
        return

    cdef:
        double distance = EPSILON
        double q_ij, tmp
        Py_ssize_t d

    # Compute squared Euclidean distance to the center of mass
    for d in range(node.n_dims):
        tmp = node.center_of_mass[d] - point[d]
        distance += tmp * tmp

    distances[i, node.num_points] = distance

    if dof <= 0:
        dof = 1e-8

    # Check if the node can be used as a summary
    if node.is_leaf or node.length / sqrt(distance) < theta:
        q_ij = 1 / (1 + distance / dof) ** dof
        qij[i, node.num_points] = q_ij * node.num_points
        sum_Q[0] += node.num_points * q_ij

        if dof != 1:
            q_ij = q_ij ** ((dof + 1) / dof)
        else:
            q_ij = q_ij * q_ij

        for d in range(node.n_dims):
            gradient[d] -= node.num_points * q_ij * (point[d] - node.center_of_mass[d])
        return

    # Recurse into child nodes
    for d in range(1 << node.n_dims):
        _estimate_negative_gradient_single(
            &node.children[d],
            point,
            gradient,
            sum_Q,
            theta,
            dof,
            distances,
            qij,
            i
        )


cdef double compute_alpha_gradient(
    double[:, ::1] p,
    double[:, :] distances,
    double[:, :] qij,
    Py_ssize_t num_points,
    double alpha,
    double sum_Q  # Pass the precomputed sum_Q
) nogil:
    """
    Compute the gradient of the loss function with respect to alpha,
    using the precomputed sum_Q for normalization.
    """
    cdef:
        Py_ssize_t i, j
        double gradient = 0.0
        double s, log_s, frac, diff, qij_norm

    # Debugging: Verify sum_Q
    #with gil:
        # print("Using Precomputed Sum_Q:", sum_Q)
        # print("Distances:", np.mean(np.asarray(distances)))
        # print("Qij:", np.mean(np.asarray(qij)))
    # Compute gradient
    for i in range(num_points):
        for j in range(num_points):
            # if i == j:
            #     continue  # Skip diagonal terms

            # Normalize qij using the precomputed sum_Q
            qij_norm = qij[i, j] / (sum_Q + EPSILON)

            # Compute terms for the gradient
            s = 1 + distances[i, j] / alpha

            log_s = log(s)
            frac = distances[i, j] / (alpha + distances[i, j])
            diff = p[i, j] - qij_norm

            # Debugging intermediate terms
            # with gil:
            #     print(f"p[{i},{j}]={p[i,j]}, qij_norm={qij_norm}, diff={diff}")
            #     print(f"s={s}, log_s={log_s}, frac={frac}")

            gradient += diff * (log_s - frac)

    # with gil:
        #print("Computed Gradient:", gradient)

    return gradient


