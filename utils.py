from ngsolve import CoefficientFunction, x, y, z
from sympy.parsing.mathematica import parse_mathematica
from sympy import printing


# FORMATTING TOOLS
class bcolors:
    """
    Class for printing in different colors.
    from https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printbf(s):
    """
    Prints string s in bold.
    Args:
        s: str
            String to be printed

    Returns:

    """
    print(f"{bcolors.BOLD}{s}{bcolors.ENDC}")
    return


# DIFFERENTIAL OPERATORS
def coef_fun_grad(u):
    """
    Computes gradient of a scalar coefficient function
    Args:
        u: CoefficientFunction

    Returns:

    """
    return CoefficientFunction(tuple([u.Diff(d) for d in [x, y, z]]))


def vec_grad(v):
    """
    Computes gradient of a (column) vector-valued Coefficient function (3d).
    Args:
        v: vector-valued Coefficient function (3d)

    Returns:
        A tensor-valued gradient Coefficient function
    """
    return CoefficientFunction(tuple([v[i].Diff(d) for i in [0, 1, 2] for d in [x, y, z]]), dims=(3, 3))


# PARSING MATHEMATICA INPUT
def sympy_to_cf(func, params):
    """
    Converts a sympy expression to a NGSolve coefficient function. Taken from
    https://ngsolve.org/forum/ngspy-forum/746-gradient-of-finite-element-function-inaccurately-integrated
    Args:
        func: str
            Mathematica expression as a string.
        params: dict
            A dictionary of parameters of the expression other than x,y,z,t.

    Returns:
        output: CoefficientFunction
            A coefficient function corresponding to the mathematica expression.
    """
    out = {}
    # print(printing.sstr(func))
    exec('from ngsolve import *; cf='+printing.sstr(func)+';', params, out)
    return out['cf']


def math_dict_to_cfs(d, params):
    """
    Takes dictionary d as an input and returns a dictionary of coefficient functions with the same keys.
    Args:
        d: dict
            A dictionary of Mathematica expressions.
        params: dict
            A dictionary of parameters of the expression other than x,y,z,t.
    Returns:
        cfs: dict
            A dictionary of corresponding coefficient functions with the same keys.
    """
    cfs = {}
    for key, value in d.items():
        # print(key, value)
        cfs[key] = sympy_to_cf(parse_mathematica(value), params)
    return cfs