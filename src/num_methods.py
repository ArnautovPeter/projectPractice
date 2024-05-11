import math
from scipy import optimize


class Solver2():
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
        xn1 = T*self._fx(xvn, yvn, *self._params) + xvn
        yn1 = T*self._fy(xvn, yvn, *self._params) + yvn
        return (xn1, yn1)
  
    """Явный метод Рунге-Кутты 4-го порядка"""
    def explicit4_method(self, T, xvn, yvn):
        kx1 = self._fx(xvn, yvn, *self._params)
        kx2 = self._fx(xvn + T/2*kx1, yvn + T/2, *self._params)
        kx3 = self._fx(xvn + T/2*kx2, yvn + T/2, *self._params)
        kx4 = self._fx(xvn + T*kx3, yvn + T, *self._params)
        xn = xvn + T/6*(kx1 + 2 * kx2 + 2 * kx3 + kx4)

        ky1 = self._fy(xvn, yvn, *self._params)
        ky2 = self._fy(xvn + T/2, yvn + T/2 * ky1, *self._params)
        ky3 = self._fy(xvn + T/2, yvn + T/2 * ky2, *self._params)
        ky4 = self._fy(xvn + T, yvn + T * ky3, *self._params)
        yn = yvn + T/6*(ky1 + 2 * ky2 + 2 * ky3 + ky4)
        
        return (xn, yn)
  
    """Явный метод Рунге-Кутты 5-го порядка"""
    def explicit5_method(self, T, xvn, yvn):
        kx1 = self._fx(xvn, yvn, *self._params)
        kx2 = self._fx(xvn + 1/4 * T * kx1,
                yvn + 3/2 * T, *self._params)
        if xvn + (3/32 * kx1 + 9/32 * kx2) * T > 10:
            print(xvn + (3/32 * kx1 + 9/32 * kx2) * T)
        kx3 = self._fx(xvn + (3/32 * kx1 + 9/32 * kx2) * T,
                yvn + 3/8 * T, *self._params)
        kx4 = self._fx(xvn + (1932/2197 * kx1 - 
                7200/2197 * kx2 + 7296/2197 * kx3) * T,
                yvn + 12/13 * T, *self._params)
        kx5 = self._fx(xvn + (439/216 * kx1 - 8 * kx2
                        + 3680/513 * kx3 - 845/4104 * kx4) * T,
                        yvn + T, *self._params)
        kx6 = self._fx(xvn + (-8/27 * kx1 + 2 * kx2 - 3544/2565 * kx3
                        + 1859/4104 * kx4 + 11/40 * kx5) * T,
                        yvn + 1/2 * T, *self._params)
        xn = xvn + (16/135 * kx1 + 6656/12825 * kx3
                    + 28561/56430 * kx4 + 2/55 * kx5
                    - 9/50 * kx6) * T
      
        ky1 = self._fy(xvn, yvn, *self._params)
        ky2 = self._fy(xvn + 3/2 * T,
                yvn + 1/4 * T * ky1, *self._params)
        ky3 = self._fy(xvn + 3/8 * T,
                yvn + (3/32 * ky1 + 9/32 * ky2) * T, *self._params)
        ky4 = self._fy(xvn + 12/13 * T, yvn + (1932/2197 * ky1 - 
                7200/2197 * ky2 + 7296/2197 * ky3) * T, *self._params)
        ky5 = self._fy(xvn + T, yvn + (439/216 * ky1 - 8 * ky2
                + 3680/513 * ky3 - 845/4104 * ky4) * T, *self._params)
        ky6 = self._fy(xvn + 1/2 * T,
                yvn + (-8/27 * ky1 + 2 * ky2 - 3544/2565 * ky3
                + 1859/4104 * ky4 + 11/40 * ky5) * T, *self._params)
        yn = yvn + (16/135 * ky1 + 6656/12825 * ky3
                    + 28561/56430 * ky4 + 2/55 * ky5
                    - 9/50 * ky6) * T

        return (xn, yn)
  

    # # НЕЯВНЫЕ МЕТОДЫ

    # """Метод Ньютона"""
    # def newtonsMethod(self, T, x, y):
    #     for _ in range(100):
    #         xn = x
    #         y_x = ((-xn+c)*T+y) / (1+b*T)  # y выраженный через х

    #         # (xn - x) / T = fx/fx'
    #         f = -xn + self._fx(xn, y_x, self._params)*T + x  # вот это есть в классе
    #         g = -1+a*T*(-T/(b*T + 1)-xn**2+1)

    #         x = x - f/g
    #     return x

    # """Неявный метод Эйлера 1-го порядка"""
    # def implicit_method(self, ):
    #     xn2 = newtonsMethod(xvn, yvn, T, self._params)
    #     yn2 = (T*(-xn2+c) + yvn)/(1+b*T)
    #     return (xn2, yn2)
        
    # def implicit2_method(self, T, xvn, yvn):
    #     # Таблица Бутчера
    #     a = [[1/4, 1/4 - math.sqrt(3)/6],
    #         [1/4 + math.sqrt(3)/6, 1/4]]
    #     c = [(1/2 - math.sqrt(3)/6), 1/2 + math.sqrt(3)/6]
    #     b = [1/2, 1/2]

    #     # x[0] = kx1, x[1] = kx2
    #     def fkx(x):
    #         xvn1 = xvn + T * (a[0][0] * x[0] + a[0][1] * x[1])
    #         xvn2 = xvn + T * (a[1][0] * x[0] + a[1][1] * x[1])
    #         # т.к. уравнения равны нулю, вычитаем kx1 и kx2
    #         return [self._fx(xvn1, yvn + c[0] * T, self._params) - x[0],
    #                 self._fx(xvn2, yvn + c[0] * T, self._params) - x[1]]
        
    #     def jac_fkx(x):
    #         xvn1 = xvn + T * (a[0][0] * x[0] + a[0][1] * x[1])
    #         xvn2 = xvn + T * (a[1][0] * x[0] + a[1][1] * x[1])
    #         # Якобиан системы, частные производные от каждого
    #         # уравнения по каждой переменной
    #         return np.array([[-ak * ((xvn1) ** 2 * T * a[0][0] - T * a[0][0]) - 1,
    #                         -ak * ((xvn1) ** 2 * T * a[0][1] - T * a[0][1])],
    #                         [-ak * ((xvn2) ** 2 * T * a[1][0] - T * a[1][0]),
    #                         -ak * ((xvn2) ** 2 * T * a[1][1] - T * a[1][1]) - 1]])


    #     sol = optimize.root(fun=fkx, x0=[0, 0],
    #                         jac=jac_fkx, method='hybr')
    #     xn = xvn + T * (b[0] * sol.x[0] + b[1] * sol.x[1])

    #     # y[0] = ky1, y[1] = ky2
    #     def fky(y):
    #         yvn1 = yvn + T * (a[0][0] * y[0] + a[0][1] * y[1])
    #         yvn2 = yvn + T * (a[1][0] * y[0] + a[1][1] * y[1])
    #         return [-(xvn + c[0] * T) - bk * yvn1 + ck - y[0],
    #                 -(xvn + c[1] * T) - bk * yvn2 + ck - y[1]]
        
    #     def jac_fky(y):
    #         return np.array([[-bk * T * a[0][0] - 1, -bk * T * a[0][1]],
    #                         [-bk * T * a[1][0], -bk * T * a[1][1] - 1]])

    #     sol = optimize.root(fun=fky, x0=[0, 0],
    #                         jac=jac_fky, method='hybr')
    #     yn = yvn + T * (b[0] * sol.x[0] + b[1] * sol.x[1])

    #     return (xn, yn)

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