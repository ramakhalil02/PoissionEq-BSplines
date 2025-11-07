import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from numpy.linalg import matrix_rank

# Constants
k = 4
Q = 1.0
delta = 1e-10
N_space = 100

#Sphere
r_min = 0.0
r_max = 2.0
R = 1.0
N_inside = 5
N_outside = 5

#Shell
R_inner = 5.0
R_outer = 10.0
R_min = 0.0
R_max = 15.0
N_shell = 4
num = 1

#Hydrogen
e = 1.602 * 1e-19
rH_min = 0.0 
rH_max = 10.0
N_H = 50


def bsplgen(x_vals, tknot, k):
    N_basis = len(tknot) - k
    n_x = len(x_vals)
    Bval = np.zeros((n_x, N_basis))
    dBval = np.zeros((n_x, N_basis))
    d2Bval = np.zeros((n_x, N_basis))

    for xi, x in enumerate(x_vals):
        B = np.zeros((len(tknot), k))
        for i in range(N_basis + k - 1):
            if tknot[i] <= x < tknot[i + 1] or (x == tknot[-1] and tknot[i + 1] == tknot[-1]):
                B[i, 0] = 1.0
        for j in range(1, k):
            for i in range(N_basis + k - j - 1):
                t1, t2 = tknot[i], tknot[i + j]
                t3, t4 = tknot[i + 1], tknot[i + j + 1]
                left = (x - t1) / (t2 - t1) * B[i, j - 1] if t2 != t1 else 0.0
                right = (t4 - x) / (t4 - t3) * B[i + 1, j - 1] if t4 != t3 else 0.0
                B[i, j] = left + right

        for i in range(N_basis):
            Bval[xi, i] = B[i, k - 1]

        for i in range(N_basis):
            if tknot[i + k - 1] != tknot[i]:
                dBval[xi, i] += (k - 1) * B[i, k - 2] / (tknot[i + k - 1] - tknot[i])
            if tknot[i + k] != tknot[i + 1]:
                dBval[xi, i] -= (k - 1) * B[i + 1, k - 2] / (tknot[i + k] - tknot[i + 1])

            term1 = term2 = term3 = term4 = 0.0
            if k > 2:
                if tknot[i + k - 1] != tknot[i] and tknot[i + k - 2] != tknot[i]:
                    term1 = B[i, k - 3] / ((tknot[i + k - 1] - tknot[i]) * (tknot[i + k - 2] - tknot[i]))
                if tknot[i + k - 1] != tknot[i] and tknot[i + k - 1] != tknot[i + 1]:
                    term2 = B[i + 1, k - 3] / ((tknot[i + k - 1] - tknot[i]) * (tknot[i + k - 1] - tknot[i + 1]))
                if tknot[i + k] != tknot[i + 1] and tknot[i + k - 1] != tknot[i + 1]:
                    term3 = B[i + 1, k - 3] / ((tknot[i + k] - tknot[i + 1]) * (tknot[i + k - 1] - tknot[i + 1]))
                if tknot[i + k] != tknot[i + 1] and tknot[i + k] != tknot[i + 2]:
                    term4 = B[i + 2, k - 3] / ((tknot[i + k] - tknot[i + 1]) * (tknot[i + k] - tknot[i + 2]))

                d2Bval[xi, i] = (k - 1) * (k - 2) * (term1 - term2 - term3 + term4)

    return Bval, dBval, d2Bval


# Uniformly charged Sphere

def knots_sphere(r_min, r_max, R, N_inside, N_outside, k, delta):
    
    r_before = np.linspace(r_min + delta, R - delta, N_inside, endpoint=False)
    
    r_after = np.linspace(R + delta, r_max - delta, N_outside, endpoint=False)
    
    r_cluster = np.array([R - delta, R, R + delta])
    
    internal_knots = np.concatenate((r_before, r_cluster, r_after))
    
    knots = np.concatenate((np.full(k, r_min), internal_knots, np.full(k, r_max)))
    return knots


def rho_sphere(r):
    return (3 * Q) / (4 * np.pi * R**3) * (r <= R)


def analytical_sphere(r_vals, R, Q):
    V = np.zeros_like(r_vals)
    for i, r in enumerate(r_vals):
        if r < R:
            V[i] = (Q / (2 * R)) * (3 - (r / R)**2)
        else:
            V[i] = Q / r
    return V


def poisson_sphere(R, Q, k, N_inside, N_outside, r_min, r_max):
    knots = knots_sphere(r_min, r_max, R, N_inside, N_outside, k, delta)
    N_basis = len(knots) - k
    N_unknowns = N_basis - 1
    N_collocation = N_unknowns - 1
    
    
    r_collocation = np.linspace(r_min + 1e-12, r_max, N_collocation)
    _, _, d2Bval = bsplgen(r_collocation, knots, k)
    A = d2Bval[:, 1:]
    B = -r_collocation * rho_sphere(r_collocation) * 4 * np.pi
    
    Bval_bc, _, _ = bsplgen([r_max], knots, k)
    A = np.vstack([A, Bval_bc[0, 1:]])
    B = np.append(B, Q)
    
    c_reduced = solve(A, B)
    
    c = np.insert(c_reduced, 0, 0)

    r_vals = np.linspace(r_min + 1e-12, r_max, N_space)
    Bval_eval, _, _ = bsplgen(r_vals, knots, k)
    phi_r = Bval_eval @ c
    V_r = phi_r / r_vals

    V_ana = analytical_sphere(r_vals, R, Q)

    plt.plot(r_vals, V_r, label="Numerical $V(r)$", linewidth=2)
    plt.plot(r_vals, V_ana, '--', label="Analytical $V(r)$", linewidth=2)
    plt.axvline(R, color='gray', linestyle='--', label="r = R")
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.title("Potential $V(r)$ from Uniformly Charged Sphere")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return r_vals, V_r, c


# Uniformly Charged Shell

def knots_shell(R_inner, R_outer, R_min, R_max, k, N_inner, N_shell, N_outer, delta):
    
    r_inner = np.linspace(R_min + delta, R_inner - delta, N_inner, endpoint=False)
    
    r_cluster_inner = np.array([R_inner - delta, R_inner, R_inner + delta])
    
    r_shell = np.linspace(R_inner + delta, R_outer - delta, N_shell, endpoint=False)

    r_cluster_outer = np.array([R_outer - delta, R_outer, R_outer + delta])
    
    r_outer = np.linspace(R_outer + delta, R_max - delta, N_outer, endpoint=False)
    
    internal_knots = np.concatenate((r_inner, r_cluster_inner, r_shell, r_cluster_outer, r_outer))
    
    knots = np.concatenate((np.full(k, R_min), internal_knots, np.full(k, R_max)))
    
    return knots



def rho_shell(r, R_inner, R_outer, Q):
    V_shell = (4/3) * np.pi * (R_outer**3 - R_inner**3)
    return (Q / V_shell) * ((r >= R_inner) & (r <= R_outer))


def analytical_shell(r_vals, Rinner, Router, Q):
    V = np.zeros_like(r_vals)
    
    for i, r in enumerate(r_vals):
        if r < Rinner:
            V[i] = ( ( 3*Q )/ ( 2 * (Router**3 - Rinner**3) )) * ( Router**2 - Rinner**2 )
        elif r <= Router:
            V[i] = ( ( 3 * Q ) / ( Router**3 - Rinner**3) ) * ( ( Router**2 / 2 ) - ( (1/3) * ( ( Rinner**3 / r ) + ( r**2 / 2 ) ) ) )
        else:
            V[i] = Q / r
    return V


def poisson_shell(R_inner, R_outer, Q, k, N_inner, N_shell, N_outer, R_min, R_max, delta):
    knots = knots_shell(R_inner, R_outer, R_min, R_max, k, N_inner, N_shell, N_outer, delta)
    N_basis = len(knots) - k
    N_unknowns = N_basis - 1
    N_collocation = N_unknowns - 1

    r_collocation = np.linspace(R_min + 1e-12 , R_max, N_collocation)
    
    _, _, d2Bval = bsplgen(r_collocation, knots, k)
    A = d2Bval[:, 1:]
    B = -r_collocation * rho_shell(r_collocation, R_inner, R_outer, Q) * 4 * np.pi

    Bval_bc, _, _ = bsplgen([R_outer-delta], knots, k)
    
    A = np.vstack([A, Bval_bc[0, 1:]])
    B = np.append(B, Q)

    c_reduced = solve(A, B)
    c = np.insert(c_reduced, 0, 0)

    r_vals = np.linspace(R_min + 1e-12, R_max, N_space)
    Bval_eval, _, _ = bsplgen(r_vals, knots, k)
    phi_r = Bval_eval @ c
    V_r = phi_r / r_vals

    V_ana = analytical_shell(r_vals, R_inner, R_outer, Q)

    plt.plot(r_vals, V_r, label="Numerical $V(r)$ (Shell)", linewidth=2)
    plt.plot(r_vals, V_ana, '--', label="Analytical $V(r)$ (Shell)", linewidth=2)
    plt.axvline(R_inner, color='gray', linestyle='--', label="Inner Radius")
    plt.axvline(R_outer, color='gray', linestyle='--', label="Outer Radius")
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.title("Potential $V(r)$ from Uniformly Charged Shell")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return r_vals, V_r, c


def rho_hydrogen(r):
    return (1 / np.pi) * np.exp(-2 * r)


def analytical_hydrogen_potential(r_vals):
    V = np.zeros_like(r_vals)
    for i, r in enumerate(r_vals):
        if r != 0:
            V[i] = (1 / r) - np.exp(-2 * r) * (1 / r + 1)
        else:
            V[i] = 0  
    return V


def poisson_hydrogen(Q, k, N_H, rH_min, rH_max, delta):
    knots = np.linspace(rH_min + delta, rH_max - delta, N_H)
    knots = np.concatenate((np.full(k, rH_min), knots, np.full(k, rH_max)))

    N_basis = len(knots) - k
    N_unknowns = N_basis - 1
    N_collocation = N_unknowns - 1

    r_collocation = np.linspace(rH_min + 1e-11, rH_max, N_collocation)
    _, _, d2Bval = bsplgen(r_collocation, knots, k)
    A = d2Bval[:, 1:]
    B = -r_collocation * rho_hydrogen(r_collocation) * 4 * np.pi

    Bval_bc, _, _ = bsplgen([rH_max - delta], knots, k)
    A = np.vstack([A, Bval_bc[0, 1:]])
    B = np.append(B, Q)

    c_reduced = solve(A, B)
    c = np.insert(c_reduced, 0, 0)

    r_vals = np.linspace(rH_min + 1e-12, rH_max, N_space)
    Bval_eval, _, _ = bsplgen(r_vals, knots, k)
    phi_r = Bval_eval @ c
    V_r = phi_r / r_vals

    V_ana = analytical_hydrogen_potential(r_vals)

    plt.plot(r_vals, V_r, label="Numerical $V(r)$ (Hydrogen)", linewidth=2)
    plt.plot(r_vals, V_ana, '--', label="Analytical $V(r)$ (Hydrogen)", linewidth=2)
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.title("Potential $V(r)$ for the Hydrogen Ground State")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    return r_vals, V_r, c


if __name__ == "__main__":
    #poisson_sphere(R, Q, k, N_inside, N_outside, r_min, r_max)
    #poisson_shell(R_inner, R_outer, Q, k, N_inside, N_shell, N_outside, R_min, R_max, delta)
    poisson_hydrogen(Q, k, N_H, rH_min, rH_max, delta)


