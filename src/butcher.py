class ButcherTable():
  # TODO: таблица используется не одинаково, например в явных
  # первая строчка пустая, а в неявных нет
  def __init__(self, a: list, b: list, c: list):
    self._a: list = a
    self._b: list = b
    self._c: list = c
    self._s: int = len(a)

  @property
  def a(self):
    return self._a
  
  @property
  def b(self):
    return self._b

  @property
  def c(self):
    return self._c
  
  @property
  def s(self):
    return self._s
