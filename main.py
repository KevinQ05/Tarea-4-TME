import numpy as np
from simplex import metodo_simplex_talegon

def main():
    c = np.array([50,60,75])
    A_eq = np.array([[1, 1, 1]])
    b_eq = np.array(475)

    bounds = [(0,160), (0,300), (50,150)]
    metodo_simplex_talegon(c, A_eq, b_eq, bounds).pretty_print(var_count=3)

if __name__ == '__main__':
    main()