import numpy as np
from simplex import metodo_simplex_talegon
from scipy.optimize import linprog

def main():
    c = np.array([50,60,75])
    A_eq = np.array([[1, 1, 1]])
    b_eq = np.array(475)
    
    A_ub = np.array([
        [1, -1, -1],
        [1, -1, 1],
    ])
    b_ub = np.array([-215, 75])

    bounds = [(0,160), (0,300), (50,150)]
    res = metodo_simplex_talegon(c, A_eq, b_eq, bounds=bounds).pretty_print(var_count=3)
    # res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    print(res)

if __name__ == '__main__':
    main()