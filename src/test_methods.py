from num_methods import Solver2
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from drawing import draw

class Test2():
    def __init__(self, f: tuple[callable, callable]):
        self._fx, self._fy = f  # точное решение
        pass

    def _slope(self, results):
        pass
    
    def test_methods(self, T_limits, step, sol: Solver2, methods: list[callable]):
        """
        T_limits - (мин шаг интегрирования, макс шаг)
        step - шаг изменения для T
        methods - численные методы которые надо проверить
        """


        TL = [T_limits[0]]
        max_error = []
        while TL[-1] <= T_limits[1]:
            results = [sol.do_method(method, TL[-1], dynamic_step=False) for method in methods]
            # args = np.linspace(T_limits[0], T_limits[1], 500)
            # # args = np.linspace(t_start, t_finish, int((t_finish - t_start) / T))
            # x_true = list([self._fx(t) for t in args])
            # draw([[i for i in results[0][0]], args], [[i[0] for i in results[0][1]], x_true], ['fake', 'real'], ['fake', 'real'])
            # return
            max_error_step = [0] * len(methods)
            tl = results[0][0]  # т.к. для всех одинаковое количество шагов
            for i in range(len(tl)):
                real_x = self._fx(tl[i])
                real_y = self._fy(tl[i])
                for j in range(len(methods)):
                    max_error_step[j] = max(max_error_step[j], abs(results[j][1][i][0] - real_x), abs(results[j][1][i][1] - real_y))
            max_error.append(max_error_step)
            TL.append(TL[-1] + step)  # один лишний будет

        # Можно сразу сдвигать и T  и f(t) но пох
        crd_move_t = math.log(T_limits[0])  # сдвиг по x
        TL = [(math.log(t) - crd_move_t) for t in TL[:-1]]
        for i in range(len(methods)):
            crd_move_f = math.log(max_error[0][i])
            # print(crd_move_f, math.log(max_error[0][i]))
            max_error_curr = [math.log(row[i]) - crd_move_f for row in max_error]

            ls = 15
            fs = 6
            fonts = 15
            fig = plt.figure(figsize=(fs,fs))
            px1 = fig.add_subplot(111)
            print(TL)
            px1.plot(TL, max_error_curr, label=methods[i].__name__ + f" {mean([(max_error_curr[i]/TL[i]) for i in range(1, len(TL))])}")

            # непонятный код))
            px1.set_xticks(np.arange(0, math.ceil(TL[-1]) + 1, step=1))
            try:
                px1.set_yticks(np.arange(0, math.ceil(max_error_curr[-1]) + 1, step=1))
            except:
                px1.set_yticks(np.arange(0, math.ceil(TL[-1]) * 9, step=5))
            # px1.set_xlabel("ln(T)")
            # px1.set_ylabel("ln(E(T))")
            px1.set_xlabel("ln(T)", fontsize = fonts)
            px1.set_ylabel("ln(E(T))", fontsize = fonts)
            px1.legend(fontsize=fonts-3)
            px1.tick_params(axis="both", which="major", labelsize=ls)
            fig.show()
                
            

    