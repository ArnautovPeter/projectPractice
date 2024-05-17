import matplotlib.pyplot as plt


"""Рисование нескольких графиков на одной, вновь созданной фигуре"""
def draw(Ox: list[list], Oy: list[list], names: list[str],
         scatter: bool = False):
    f = plt.figure(figsize=(10, 7))
    px1 = f.add_subplot(111)
    if not scatter:
        for x, y, name in zip(Ox, Oy, names):
            px1.plot(x, y, label=name)
    else:
        for x, y in zip(Ox, Oy, names):
            px1.scatter(x, y, label=name)
    px1.legend()
    f.show()


"Рисование графиков"
def draw_on_plot(Ox: list[list], Oy: list[list], names: list[str],
                 px1, scatter: bool = False):
    if not scatter:
        for x, y, name in zip(Ox, Oy, names):
            px1.plot(x, y, label=name)
    else:
        for x, y in zip(Ox, Oy, names):
            px1.scatter(x, y, label=name)

"Рисование графиков"
def draw_on_plot_label(Ox: list[list], Oy: list[list], names: list[str],
                 px1, xlable, ylable, scatter: bool = False):
    if not scatter:
        for x, y, name in zip(Ox, Oy, names):
            px1.plot(x, y, label=name)
    else:
        for x, y in zip(Ox, Oy, names):
            px1.scatter(x, y, label=name)

