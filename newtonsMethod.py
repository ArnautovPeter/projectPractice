from sympy import *
import math
T = Symbol("T")
x = Symbol("x")
a = Symbol("a")
b = Symbol("b")
xn = Symbol("xn")
y = Symbol("y")
c = Symbol("c")
f = -xn+(-a*(xn**3/3-xn)+a*((-xn+c)*T+y)/(1+b*T))*T+x
g = -1+a*T*(-T/(b*T+1)-xn**2+1)
l = [0.5]
for i in range(13):
    l.append(l[-1] - f.subs([(xn, l[-1]), (x, 2), (y, 0), (b, 1.2), (c, 0.5), (a, 100), (T, 0.001)]) / \
             g.subs([(xn, l[-1]), (x, 2), (y, 0), (b, 1.2), (c, 0.5), (a, 100), (T, 0.001)]))
print(l)