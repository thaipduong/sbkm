import numpy as np
# Solve this equation: (a1*t^2 + b1*t + c1 - x1)**2 + (a2*t^2 + b2*t + c2 - x2)**2 = r**2
# x1 = s1(curr_t), x2 = s2(curr_t)
def solve_next_t(a1, b1, c1, a2, b2, c2, curr_t, r):
    x1 = a1*curr_t**2 + b1*curr_t + c1
    x2 = a2 * curr_t ** 2 + b2 * curr_t + c2
    p = []
    p.append(a1**2 + a2**2)
    p.append(2*a1*b1 + 2*a2*b2)
    p.append(b1**2 + 2*a1*(c1-x1) + b2**2 + 2*a2*(c2-x2))
    p.append(2*b1*(c1-x1) + 2*b2*(c2-x2))
    p.append((c1-x1)**2 + (c2-x2)**2 - r**2)
    roots = np.roots(p)
    real_roots = roots.real[abs(roots.imag) < 1e-6]
    min_t = -1
    for root in real_roots:
        if root > curr_t:
            if min_t < 0 or root < min_t:
                min_t = root
    return min_t