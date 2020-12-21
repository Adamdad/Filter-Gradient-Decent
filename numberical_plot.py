from numpy import exp, arange
from matplotlib.pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
import numpy as np
import pylab
import json


# the function that I'm going to plot
def z_func(x, y):
    return (x ** 2 + y ** 2) + 5 * np.sin(x + y ** 2)


record = [
    ['record/v2/numerical/NonConvex_Adam_0.03.json', 'Adam'],
    ['record/v2/numerical/NonConvex_FGD-K_0.03.json', 'FGD-K'],
    ['record/v2/numerical/NonConvex_FGD-W_0.03.json', 'FGD-W'],
    ['record/v2/numerical/NonConvex_SGD_0.03.json', 'SGD'],
    ['record/v2/numerical/NonConvex_ARMAGD_0.03_[0, 0.9].json', 'FGD-AR(1)'],
    ['record/v2/numerical/NonConvex_ARMAGD_0.03_[0.1, 0.8].json', 'FGD-AR(2)'],
    ['record/v2/numerical/NonConvex_MASGD_0.03_[0.0, 0.9].json', 'FGD-MA(1)'],
    ['record/v2/numerical/NonConvex_MASGD_0.03_[0.1, 0.8].json', 'FGD-MA(2)']
]
x = arange(-6.0, 6.1, 0.1)
y = arange(-6.0, 6.1, 0.1)
X, Y = np.meshgrid(x, y, sparse=True)  # grid of point
Z = z_func(X, Y)  # evaluation of the function on the grid

im = imshow(Z, extent=[-6, 6, -6, 6], cmap=cm.RdBu)  # drawing the function
# im = pylab.contourf(x, y, Z, cmap=cm.RdBu)
# adding the Contour lines with labels
cset = contour(x, y, Z, linewidths=1, cmap=cm.Set2)
clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
colorbar(im)  # adding the colobar on the right

# for item in record:
#     with open(item[0], "r") as read_file:
#         data = json.load(read_file)
#     X = np.array(data['x'])
#     Y = np.array(data['y'])
#
#     pylab.plot(X, Y, label=item[1], alpha=0.8)


pylab.xlabel('x')
pylab.ylabel('y')
pylab.xlim(-6, 6)
pylab.ylim(-6, 6)
# pylab.legend()
# latex fashion title
title('$f(x,y)=(x^2+y^2) + \sin{(x+2y^2)}$')

show()
