# Truncation Error Analysis of the 5-Point Laplacian

## Introduction

This report derives the truncation error of the 5-point finite difference approximation
to the Laplacian operator, demonstrating that it is second-order accurate on a uniform grid.

## The Laplace Equation

We solve the 2D Laplace equation for the stream function psi(x,y):

```
d2(psi)/dx2 + d2(psi)/dy2 = 0
```

This equation arises from:
- Stream function definition: u = d(psi)/dy, v = -d(psi)/dx
- Irrotational flow condition: vorticity = 0

## 5-Point Finite Difference Stencil

On a uniform grid with spacing dx in x-direction and dy in y-direction, the discrete
Laplacian at node (i,j) is:

```
L_h[psi]_{i,j} = (psi_{i+1,j} - 2*psi_{i,j} + psi_{i-1,j}) / dx^2
               + (psi_{i,j+1} - 2*psi_{i,j} + psi_{i,j-1}) / dy^2
```

For dx = dy = h, this simplifies to:

```
L_h[psi]_{i,j} = (psi_{i+1,j} + psi_{i-1,j} + psi_{i,j+1} + psi_{i,j-1} - 4*psi_{i,j}) / h^2
```

## Taylor Series Expansion

To derive the truncation error, we expand the neighboring values using Taylor series
around the point (x_i, y_j).

### Expansion in x-direction

Let psi(x,y) be sufficiently smooth. Expanding psi_{i+1,j} = psi(x_i + dx, y_j):

```
psi_{i+1,j} = psi_{i,j} 
            + dx * (d psi/dx)_{i,j}
            + (dx^2/2) * (d2 psi/dx2)_{i,j}
            + (dx^3/6) * (d3 psi/dx3)_{i,j}
            + (dx^4/24) * (d4 psi/dx4)_{i,j}
            + O(dx^5)
```

Similarly, expanding psi_{i-1,j} = psi(x_i - dx, y_j):

```
psi_{i-1,j} = psi_{i,j}
            - dx * (d psi/dx)_{i,j}
            + (dx^2/2) * (d2 psi/dx2)_{i,j}
            - (dx^3/6) * (d3 psi/dx3)_{i,j}
            + (dx^4/24) * (d4 psi/dx4)_{i,j}
            + O(dx^5)
```

### Sum of x-neighbors

Adding psi_{i+1,j} and psi_{i-1,j}:

```
psi_{i+1,j} + psi_{i-1,j} = 2*psi_{i,j}
                          + dx^2 * (d2 psi/dx2)_{i,j}
                          + (dx^4/12) * (d4 psi/dx4)_{i,j}
                          + O(dx^6)
```

Note: Odd-order terms cancel due to symmetry.

### Second derivative approximation in x

Rearranging:

```
(psi_{i+1,j} - 2*psi_{i,j} + psi_{i-1,j}) / dx^2 = (d2 psi/dx2)_{i,j}
                                                  + (dx^2/12) * (d4 psi/dx4)_{i,j}
                                                  + O(dx^4)
```

The leading truncation error term is:

```
T_x = (dx^2/12) * (d4 psi/dx4)_{i,j}
```

### Expansion in y-direction

By the same procedure for y-direction:

```
psi_{i,j+1} + psi_{i,j-1} = 2*psi_{i,j}
                          + dy^2 * (d2 psi/dy2)_{i,j}
                          + (dy^4/12) * (d4 psi/dy4)_{i,j}
                          + O(dy^6)
```

Leading to:

```
(psi_{i,j+1} - 2*psi_{i,j} + psi_{i,j-1}) / dy^2 = (d2 psi/dy2)_{i,j}
                                                  + (dy^2/12) * (d4 psi/dy4)_{i,j}
                                                  + O(dy^4)
```

The leading truncation error term is:

```
T_y = (dy^2/12) * (d4 psi/dy4)_{i,j}
```

## Total Truncation Error

Combining the x and y approximations, the discrete Laplacian is:

```
L_h[psi]_{i,j} = (d2 psi/dx2 + d2 psi/dy2)_{i,j}
               + (dx^2/12) * (d4 psi/dx4)_{i,j}
               + (dy^2/12) * (d4 psi/dy4)_{i,j}
               + O(dx^4, dy^4)
```

The truncation error of the 5-point Laplacian is:

```
tau_{i,j} = L_h[psi]_{i,j} - (d2 psi/dx2 + d2 psi/dy2)_{i,j}
          = (dx^2/12) * (d4 psi/dx4)_{i,j} + (dy^2/12) * (d4 psi/dy4)_{i,j}
          + O(dx^4, dy^4)
```

## Order of Accuracy

The truncation error is:

```
tau = O(dx^2) + O(dy^2) = O(dx^2 + dy^2)
```

This shows that the 5-point finite difference approximation to the Laplacian is
**second-order accurate** in both spatial directions.

### Uniform Grid Case (dx = dy = h)

For a uniform grid with h = dx = dy:

```
tau = (h^2/12) * (d4 psi/dx4 + d4 psi/dy4) + O(h^4)
    = O(h^2)
```

## Implications for Solution Accuracy

The truncation error analysis tells us:

1. **Grid refinement**: Halving the grid spacing (h -> h/2) reduces the truncation
   error by a factor of 4 (since error ~ h^2).

2. **Convergence rate**: As we refine the grid, the numerical solution converges
   to the exact solution at a rate proportional to h^2.

3. **Smoothness requirement**: The error analysis assumes psi has continuous
   fourth derivatives. Near singularities or discontinuities, the actual error
   may be larger.

## Verification

The second-order accuracy can be verified numerically by:

1. Solving on successively refined grids (h, h/2, h/4, ...)
2. Computing the error using a reference solution or Richardson extrapolation
3. Confirming that the error ratio between successive refinements approaches 4

For our problem with dx = dy = 0.1 on a 101x101 grid, the expected truncation
error magnitude is approximately:

```
|tau| ~ (0.1)^2 / 12 * |d4 psi/dx4 + d4 psi/dy4|
      ~ 0.00083 * (fourth derivatives)
```

## Summary

The 5-point finite difference stencil for the Laplacian operator:

```
L_h[psi] = (psi_{i+1,j} + psi_{i-1,j} + psi_{i,j+1} + psi_{i,j-1} - 4*psi_{i,j}) / h^2
```

has a truncation error of O(h^2), making it a second-order accurate scheme.
This is sufficient for most engineering applications and provides a good balance
between accuracy and computational cost.

## References

1. LeVeque, R.J. (2007). Finite Difference Methods for Ordinary and Partial
   Differential Equations. SIAM.

2. Ferziger, J.H. and Peric, M. (2002). Computational Methods for Fluid Dynamics.
   Springer.

3. Anderson, J.D. (1995). Computational Fluid Dynamics: The Basics with Applications.
   McGraw-Hill.
