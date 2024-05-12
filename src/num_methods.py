import math
from scipy import optimize


class Solver2():
    # не сработает если есть t, т.е. независимая переменная
    def __init__(self, func: tuple[callable, callable],
                params: list[float],
                init_val: tuple[float, float],
                t_limits: tuple[float, float]) -> None:
        # неизменные поля для Solver
        # у функций должно быть ОДИНАКОВОЕ кол-во параметров
        self._fx = func[0]
        self._fy = func[1]
        self._init_val = init_val

        # Поля, которые можно менять для изучения поведения
        self._t_limits = t_limits
        self._params = params

    """Изменение параметров уравнения"""
    def set_params(self, new_params):
        self._params = new_params

    """Изменение диапазона интегрирования"""
    def set_t_limits(self, new_limits):
        self._t_limits = new_limits

    # ЯВНЫЕ МЕТОДЫ

    """Явный метод Эйлера 1-го порядка"""
    def explicit1_method(self, T, xvn, yvn):
        return self._runge_kutta_exp(T, xvn, yvn, [], [1], [0])
  
    """Явный метод Рунге-Кутты 4-го порядка"""
    def explicit4_method(self, T, xvn, yvn):
        a = [[1/2],
             [0, 1/2],
             [0, 0, 1]]
        b = [1/6, 2/6, 2/6, 1/6]
        c = []
        return self._runge_kutta_exp(T, xvn, yvn, a, b, c)
  
    """Явный метод Рунге-Кутты 5-го порядка"""
    def explicit5_method(self, T, xvn, yvn):
        a = [[1/4],
             [3/32, 9/32],
             [1932/2197, -7200/2197, 7296/2197],
             [439/216, -8, 3680/513, -845/4104],
             [-8/27, 2, -3544/2565, 1859/4104, -11/40]]
        b = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
        c = []
        return self._runge_kutta_exp(T, xvn, yvn, a, b)
    
    def _runge_kutta_exp(self, T, xvn, yvn, a, b, c):
        kx = [T * self._fx(T, xvn, yvn, *self._params)]
        ky = [T * self._fy(T, xvn, yvn, *self._params)]

        for i in range(len(a)):
            to_pass = (T * c[i],
                xvn + sum([a[i][j] * kx[j] for j in range(i + 1)]),
                       yvn + sum([a[i][j] * ky[j] for j in range(i + 1)]),
                       *self._params)
            kx.append(T * self._fx(*to_pass))
            ky.append(T * self._fy(*to_pass))
        
        xn = xvn + sum([b[i] * kx[i] for i in range(len(kx))])
        yn = yvn + sum([b[i] * ky[i] for i in range(len(ky))])

        return (xn, yn)

    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ

    # новый шаг
    def get_p(self, array):
        s = sum([i ** 2 for i in array])
        return math.sqrt(s)

    # T0 - максимальный возможный шаг
    def new_T(self, last_T, T0, array, array_prev, L):
        p = self.get_p(array)
        F = [i / p for i in array]
        p_prev = self.get_p(array_prev)
        F_prev = [i / p_prev for i in array_prev]
        X = [(F[i] - F_prev[i]) / last_T for i in range(len(F))]
        XX = sum([i ** 2 for i in X])
        new_T = T0 / (1 + math.sqrt(L) * XX ** (1/4))
        return new_T

    """
    Выполнение метода
    method - метод, которым надо решить
    T0 - максимальный шаг интегрирования, если динам., иначе просто шаг
    dynamic_step - использовать ли динамический шаг
    """
    def do_method(self, method, T0, dynamic_step: bool=True) -> tuple[list[float],
                                                                    list[tuple[float, float]],
                                                                    list[tuple[float, float]],
                                                                    str]:
        tl = [self._t_limits[0]]
        T = T0

        # список переменных в виде (x, y)
        array: list[tuple] = [self._init_val]

        array_T = [T]
        array_dif = [(self._fx(array[-1][0], array[-1][1], *self._params),
                        self._fy(array[-1][0], array[-1][1], *self._params))]
        while tl[-1] <= self._t_limits[1]:
            array.append(method(T, array[-1][0], array[-1][1]))
            array_dif.append((self._fx(array[-1][0], array[-1][1], *self._params),
                    self._fy(array[-1][0], array[-1][1], *self._params)))
            if dynamic_step:
                T = self.new_T(array_T[-1], T0, array[-1], array[-2],
                                self._t_limits[1] - self._t_limits[0])
            array_T.append(T)
            tl.append(tl[-1] + T)
        return (tl, array, array_dif, method.__name__)
