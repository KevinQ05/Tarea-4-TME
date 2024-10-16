import numpy as np
from dataclasses import dataclass

np.seterr(divide="ignore")
Array = np.ndarray


def switch_array_columns(array: Array, i1: int, i2: int) -> Array:
    new_array = array.copy()
    match new_array.shape:
        case (_,):
            new_array[[i1, i2]] = new_array[[i2, i1]]

        case (_, _):
            new_array[:, [i1, i2]] = new_array[:, [i2, i1]]

        case _:
            raise (ValueError)

    return new_array


def switch_array_rows(array: Array, i1: int, i2: int) -> Array:
    new_array = array.copy()
    new_array[[i1, i2]] = new_array[[i2, i1]]

    return new_array


@dataclass
class IterationResult:
    z: float
    x: Array
    pi: Array
    c_reducidos: Array
    b_barra: Array
    Y: Array
    order: Array

    def pretty_print(self, var_count: int = None):
        if var_count == None:
            var_count = len(self.x)

        variable_values = {f"x{x}": self.x[idx, 0]
                           for idx, x in np.ndenumerate(self.order)}
        sorted_variable_values = {k: v for k, v in sorted(
            variable_values.items(), key=lambda item: item[0])}

        print_str = f'Funcion objetivo: z = {self.z[0, 0]} \n'
        for idx, (key, val) in enumerate(sorted_variable_values.items()):
            if idx == var_count:
                break
            print_str += f'{key} = {val} \n'
        print_str += f"Costos reducidos: {self.c_reducidos}"

        print(print_str)


def iteracion_simplex(c: Array, A: Array, b: Array, order: Array | None = None) -> IterationResult:
    if len(c.shape) == 1:
        raise ValueError("C debe ser un vector columna")
    if len(b.shape) == 1:
        raise ValueError("b debe ser un vector columna")

    m, n = A.shape

    # El orden se utiliza para no perder las variables, es decir, como ID de cada variable (x0, x1, x2, ... xn)
    if order is None:
        new_order = np.arange(0, n)
    else:
        new_order = order

    # Separar matriz A en variables básicas y no básicas
    B = A.copy()[:, 0:m]
    N = A.copy()[:, m:n]

    # Costos c básicos y no básicos
    c_B = c.copy()[0:m, 0:]
    c_N = c.copy()[m:n, 0:]

    # La matriz B podría o no ser singular, hay que compensar por ese caso
    try:
        B_inv = np.linalg.inv(B)
    except Exception as _:
        i1 = np.random.randint(0, n)
        i2 = np.random.randint(0, n)

        A_new = switch_array_columns(A, i1, i2)
        c_new = switch_array_rows(c, i1, i2)
        new_order = switch_array_columns(new_order, i1, i2)
        return iteracion_simplex(c_new, A_new, b, new_order)

    # solución inicial z0 = z(x0)
    x0 = B_inv@b
    z0 = (c_B.T)@x0

    # Si alguna xn inicial es negativa, no es una solución factible, volver a empezar
    if any(x < 0 for x in x0):
        i1 = np.random.randint(0, n)
        i2 = np.random.randint(0, n)

        A_new = switch_array_columns(A, i1, i2)
        c_new = switch_array_rows(c, i1, i2)
        new_order = switch_array_columns(new_order, i1, i2)
        return iteracion_simplex(c_new, A_new, b, new_order)

    # Variables duales
    pi = c_B.T@B_inv
    costos_reducidos = np.subtract(pi@N, c_N.T)

    # Necesario a posterior para decidir cuales variables intercambiar
    b_barra = B_inv@b
    Y = B_inv@N

    # x0 incluyendo las variables no básicas, que son cero
    result_x = np.zeros((n, 1))
    result_x[0:m, :] = x0

    res = {
        "z": z0,
        "x": result_x,
        "pi": pi,
        "c_reducidos": costos_reducidos,
        "b_barra": b_barra,
        "Y": Y,
        "order": new_order,
    }

    res = IterationResult(*res.values())
    return res

# Toma las decisiones de cambio de variables


def metodo_simplex_revisado(c: Array, A: Array, b: Array, order: Array | None = None, verbose: bool = False, iter: int = 0) -> IterationResult:
    m, _ = A.shape

    inicial = iteracion_simplex(c, A, b, order)
    c_reducidos = inicial.c_reducidos

    # Detenerse cuando todos los valores de costos reducidos sean negativos o cero
    if all(x <= 0 for x in c_reducidos[0]):
        return inicial

    # La variable que entra a la base es la de mayor valor positivo
    indice_no_basicas = np.argmax(c_reducidos)
    Y = inicial.Y
    b_barra = inicial.b_barra
    cocientes = np.divide(b_barra, np.vstack(Y[:, indice_no_basicas]))

    # La variable que sale de la base es la de cociente menor positivo
    indice_basicas = np.where(cocientes > 0, cocientes, np.inf).argmin()

    i1, i2 = indice_basicas, indice_no_basicas + m

    A_new = switch_array_columns(A, i1, i2)
    c_new = switch_array_rows(c, i1, i2)
    order_new = switch_array_columns(order.reshape(1, -1), i1, i2).flatten()

    # Hacky print statement to show iterations
    if verbose:
        print(f"Iteracion {iter}:")
        inicial.pretty_print()
        print("A = ")
        # First print the variable names (headers for each column)
        for idx in order:
            print(f"x{idx}", end="\t")

        print()  # Newline after printing headers

        # Now print the matrix A with each column under its corresponding variable name
        for row in A:
            for element in row:
                print(f"{element}", end="\t")
            print()  # Newline after each row
        print(f"Cocientes b/Y = \n{cocientes}")
        print(f"x{order[i1]} sale de la base, x{order[i2]} entra a la base")

    return metodo_simplex_revisado(c_new, A_new, b, order_new, verbose=verbose, iter=iter+1)


def build_standard_matrices(c: Array, A_eq: Array, b_eq: Array, A_ub=None, b_ub=None, bounds=list[tuple]) -> dict:
    # Inspirado en scipy.optimize.linprog, toma un vector de costos c, una matrix de igualdades tales que A_eq@x = b_eq
    # además de una lista de tuplas (inferior, superior) con los límites de cada variable x
    # No es necesario incluir variables de holgura, pues lo hace automáticamente

    eq_rows, eq_colums = A_eq.shape
    if isinstance(A_ub, Array):
        ineq_rows, ineq_columns = A_ub.shape
    else:
        ineq_rows, ineq_columns = (0, 0)

    # Separar limites en inferiores y superiores siempre que sean mayores a cero
    lower_bounds, upper_bounds = zip(*bounds)
    lower_bounds = list(filter(lambda x: x > 0, lower_bounds))
    upper_bounds = list(filter(lambda x: x > 0, upper_bounds))

    upper_count = len(upper_bounds)
    lower_count = len(lower_bounds)
    rows = eq_rows + upper_count + lower_count + ineq_rows
    cols = eq_colums + upper_count + lower_count + ineq_columns

    slack_variable_count = cols - eq_colums
    normal_variable_count = eq_colums

    A = np.zeros(shape=(rows, cols))
    # equalities
    A[0:eq_rows, cols-eq_colums:cols] = A_eq
    # Fill in basic variables last
    A[eq_rows:eq_rows+eq_colums, cols -
        eq_colums:cols] = np.eye(normal_variable_count)
    # then slack upper limits
    A[eq_rows:eq_rows+eq_colums, 0:eq_colums] = np.eye(normal_variable_count)

    lower_bound_count = 0

    # Añadir a la matriz A las restricciones de los límites
    for idx, (low, _) in enumerate(bounds):
        if low > 0:
            A[rows-1-lower_bound_count, cols -
                (normal_variable_count - idx)] = 1
            A[rows-1-lower_bound_count, slack_variable_count-lower_bound_count-1] = -1
            lower_bound_count += 1

    if A_ub is not None:
        A[eq_rows+eq_colums:eq_rows+eq_colums +
            ineq_rows, cols-ineq_columns:cols] = A_ub
        for idx in range(ineq_rows):
            A[eq_rows+eq_colums+idx, eq_colums + idx] = 1

    c_new = np.zeros(shape=(cols, 1))
    c_new[cols-eq_colums: cols, 0] = c

    b_new = np.zeros(shape=(rows, 1))
    b_new[0:eq_rows] = b_eq
    b_new[eq_rows:eq_rows+upper_count] = np.vstack(upper_bounds)
    if b_ub is not None:
        b_new[eq_rows+upper_count:eq_rows +
              upper_count+ineq_rows] = np.vstack(b_ub)
    b_new[rows -
          lower_bound_count:rows] = np.vstack([x for x in reversed(lower_bounds)])
    return {
        'c': c_new,
        'A': A,
        'b': b_new,
        'basic_vars': normal_variable_count,
        'slack_vars': slack_variable_count
    }


def metodo_simplex_talegon(c: Array, A_eq: Array, b_eq: Array, A_ub=None, b_ub=None, bounds=list[tuple], verbose: bool = False) -> IterationResult:
    # Se encarga de pasar de una forma amigable al usuario a la forma estándar
    standard_form = build_standard_matrices(
        c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    c_std = standard_form['c']
    A_std = standard_form['A']
    b_std = standard_form['b']

    basic_vars = standard_form['basic_vars']
    slack_vars = standard_form['slack_vars']

    order = [x for x in range(basic_vars, slack_vars+basic_vars)]
    order += [x for x in range(0, basic_vars)]
    order = np.array(order)

    result = metodo_simplex_revisado(
        c_std, A_std, b_std, order, verbose=verbose)

    return result
