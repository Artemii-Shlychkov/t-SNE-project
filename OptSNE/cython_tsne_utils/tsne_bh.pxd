# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3

cimport numpy as cnp
from cython cimport Py_ssize_t
from quad_tree cimport QuadTree, Node

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
)


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
) nogil

cdef double compute_alpha_gradient(
    double[:, ::1] p,
    double[:, :] distances,
    double[:, :] qij,
    Py_ssize_t num_points,
    double alpha,
    double sum_Q
) nogil
