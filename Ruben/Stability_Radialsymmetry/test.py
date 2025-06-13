import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def D1_zeta(n, h):
    """
    Baut die (n×n)-Matrix für ∂ζ/∂r, wobei an den "zweiten" Punkten (i=1 und i=n-2)
    jeweils ein Vorwärts- bzw. Rückwärts-Stencil benutzt wird, und in den inneren Punkten
    (i=2..n-3) die zentrale Differenz. An den Rändern (i=0, i=n-1) bleiben die Zeilen 0,
    weil D0_zeta dort ζ=0 bzw. ζ(1)=0 / ζ(n-2)=0 bereits erzwingt.
    """
    D1 = lil_matrix((n, n))

    # --------------------------
    # 1) i = 0  (Gradient wird hier über D0_zeta auf 0 gesetzt)
    #    → ganze Zeile = 0 (bleibt so)
    # --------------------------

    # --------------------------
    # 2) i = 1  (untere Grenze: ζ'(x0) ≈ (ζ(x2)-ζ(x1)) / h,
    #    denn ζ(x0)=0 per Dirichlet)
    #    → (ζ[2] - ζ[1]) / h = 0
    #    Zeile i=1:  -1/h * ζ[1]  + 1/h * ζ[2]
    # --------------------------
    D1[1, 1] = -1.0 / h
    D1[1, 2] =  1.0 / h

    # --------------------------
    # 3) i = 2..n-3: zentrales Stencil
    #    ζ'(r_i) ≈ ( ζ[i+1] - ζ[i-1] ) / (2h)
    # --------------------------
    for i in range(2, n-2):
        D1[i, i-1] = -1.0 / (2*h)
        D1[i, i+1] = +1.0 / (2*h)
        # D1[i, i] = 0 (bleibt Null)

    # --------------------------
    # 4) i = n-2  (obere Grenze: ζ'(x_{n-1}) ≈ (ζ[n-2] - ζ[n-3]) / h,
    #    denn ζ[n-1] = 0 per Dirichlet)
    #    Zeile i=n-2:  +1/h * ζ[n-2]  - 1/h * ζ[n-3]
    # --------------------------
    D1[n-2, n-3] = -1.0 / h
    D1[n-2, n-2] = +1.0 / h

    # --------------------------
    # 5) i = n-1  (Gradient wird hier über D0_zeta auf 0 gesetzt)
    #    → ganze Zeile = 0 (bleibt so)
    # --------------------------

    return D1.tocsr()
A = D1_zeta(5, 1)
print(A)
n=10
for i in range(1,n-1):
    print(i)
