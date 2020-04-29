# %%
"""
Übung 2.1
*********

Visualisierung eindimensionaler Arrays


"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.001)
y = x ** 2
z = x * 10

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, z)


# %%
"""
Übung 2.2
*********
Visualisierung multipler eindimensionaler Arrays

"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.001)

fig, axes = plt.subplots(4, 2)
for idx, ax in enumerate(axes.flatten()):
    ax.plot(x, x ** idx)
# %%
"""
Übung 2.3
*********

Visualisierung von 2-dimensionalen Arrays und Annotation

"""
xlen = 10
ylen = 9

x = np.reshape(np.arange(0, xlen * ylen), (ylen, xlen))
fig, ax = plt.subplots()
im = ax.imshow(x)
ax.set_xticks(range(xlen))
ax.set_yticks(range(ylen))
ax.set_yticklabels(list("ABCDEFGHI"))
fig.colorbar(im)

# %%
"""
Übung 2.4
*********

Beispiel mit echten Daten

"""
import numpy as np
import json

rs = dict()
with np.load("resting_state.npz") as f:
    for key in f.keys():
        rs[key] = f[key]
with open("resting_state.json") as f:
    rs.update(**json.loads(f.read()))
