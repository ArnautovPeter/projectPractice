import math
from scipy import optimize
from butcher import ButcherTable


class Solver2():
    expl1_table = ButcherTable([], [1], [])
    expl4_table = ButcherTable([[1/2],
                                [0, 1/2],
                                [0, 0, 1]],
                                [1/6, 2/6, 2/6, 1/6],
                                [1/2, 1/2, 1])
    expl5_table = ButcherTable([[1/4],
                                [3/32, 9/32],
                                [1932/2197, -7200/2197, 7296/2197],
                                [439/216, -8, 3680/513, -845/4104],
                                [-8/27, 2, -3544/2565, 1859/4104, -11/40]],
                                [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],
                                [1/4, 3/8, 12/13, 1, 1/2])
    impl1_table = ButcherTable([],
                               [],
                               [])
    
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

    def set_params(self, new_params):
        """Изменение параметров уравнения"""    
        self._params = new_params

    def set_t_limits(self, new_limits):
        """Изменение диапазона интегрирования"""
        self._t_limits = new_limits

    # ЯВНЫЕ МЕТОДЫ

    def explicit1(self, T, xvn, yvn, t, _):
        """Явный метод Эйлера 1-го порядка"""
        return self._runge_kutta_exp(T, xvn, yvn, t, Solver2.expl1_table)
  
    def explicit4(self, T, xvn, yvn, t, _):
        """Явный метод Рунге-Кутты 4-го порядка"""
        return self._runge_kutta_exp(T, xvn, yvn, t, Solver2.expl4_table)
  
    def explicit5(self, T, xvn, yvn, t, _):
        """Явный метод Рунге-Кутты 5-го порядка"""
        return self._runge_kutta_exp(T, xvn, yvn, t, Solver2.expl5_table)
    
    def _runge_kutta_exp(self, T: float, xvn: float, yvn: float, t: float, table: ButcherTable):
        kx = [T * self._fx(t, xvn, yvn, *self._params)]
        ky = [T * self._fy(t, xvn, yvn, *self._params)]

        for i in range(table.s):
            to_pass = (t + T * table.c[i],
                       xvn + sum([table.a[i][j] * kx[j] for j in range(i + 1)]),
                       yvn + sum([table.a[i][j] * ky[j] for j in range(i + 1)]),
                       *self._params)
            kx.append(T * self._fx(*to_pass))
            ky.append(T * self._fy(*to_pass))
        
        xn = xvn + sum([table.b[i] * kx[i] for i in range(len(kx))])
        yn = yvn + sum([table.b[i] * ky[i] for i in range(len(ky))])

        return (xn, yn)
    
    # НЕЯВНЫЕ МЕТОДЫ

    def _guess(self, T, xvn, yvn, t):
        # TODO: maybe xvn is better, needs to be checked
        return [xvn + T * self._fx(T, xvn, yvn, t),
                yvn + T * self._fy(T, xvn, yvn, t)]
    
    def _sol_func(self, T, xvn, yvn, t):
        def f(x):
            return [T * self._fx(t, x[0], x[1], *self._params) - x[0] + xvn,
                    T * self._fy(t, x[0], x[1], *self._params) - x[1] + yvn]
        return f
    
    def implicit1(self, T, xvn, yvn, t, jac):
        return self._runge_kutta_imp(self, T, xvn, yvn, t, jac)
    
    def _runge_kutta_imp(self, T, xvn, yvn, t, table, jac):
        # print(jac(T, xvn, yvn, t, *self._params))
        sol = optimize.root(fun=self._sol_func(T, xvn, yvn, t),
                            x0=self._guess(T, xvn, yvn, t),
                            jac=jac(T, xvn, yvn, t, *self._params),
                            method='hybr')
        return (sol.x[0], sol.x[1])

    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ

    def get_p(self, array):
        """ новый шаг """
        s = sum([i ** 2 for i in array])
        return math.sqrt(s)

    def new_T(self, last_T, T0, array, array_prev, L):
        """ T0 - максимальный возможный шаг """
        p = self.get_p(array)
        F = [i / p for i in array]
        p_prev = self.get_p(array_prev)
        F_prev = [i / p_prev for i in array_prev]
        X = [(F[i] - F_prev[i]) / last_T for i in range(len(F))]
        XX = sum([i ** 2 for i in X])
        new_T = T0 / (1 + math.sqrt(L) * XX ** (1/4))
        return new_T

    def do_method(self, method, T0, jac_impl: callable=None, dynamic_step: bool=False) -> tuple[list[float],
                                                                                                list[tuple[float, float]],
                                                                                                list[tuple[float, float]],
                                                                                                str]:
        """
        Выполнение численного метода\n
        method - метод, которым надо решить\n
        T0 - максимальный шаг интегрирования, если динам., иначе просто шаг\n
        якобиан системы уравнений, которая является решением для неявной схемы, передается в виде
        jac(T, xvn, yvn, t, params) и содержит внутри функцию, которую нужно передать optimize.root\n
        dynamic_step - использовать ли динамический шаг

        -----
        
        return ([независимая переменная], {вычисленные значения для ф-ий}, [названия методов])
        """
        
        tl = [self._t_limits[0]]
        T = T0

        # список переменных в виде (x, y)
        array: list[tuple] = [self._init_val]
        array_T = [T]
        while tl[-1] <= self._t_limits[1]:
            array.append(method(T, array[-1][0], array[-1][1], tl[-1], jac_impl))
            if dynamic_step:
                T = self.new_T(array_T[-1], T0, array[-1], array[-2],
                                self._t_limits[1] - self._t_limits[0])
            array_T.append(T)
            tl.append(tl[-1] + T)
        return (tl, array, method.__name__)
    
    @staticmethod
    def get_x(array):
        """Получение первой координаты из кортежа от do_method"""
        return [i[0] for i in array[1]]

    @staticmethod
    def get_y(array):
        """Получение второй координаты из кортежа от do_method"""
        return [i[1] for i in array[1]]
    
    def get_diffs(self):
        array_dif = [(self._fx(tl[-1], array[-1][0], array[-1][1], *self._params),
                        self._fy(tl[-1], array[-1][0], array[-1][1], *self._params))]
    