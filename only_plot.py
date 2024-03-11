import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.FFM import FloorFieldModel

model = FloorFieldModel(r"map/Umeda_underground.npy", SFF=True, method="Linf")
# model = FloorFieldModel(r"map/Umeda.npy", SFF=True, method="Linf")
model.paraname = "ks3_kd1"
model.number = 0
model.dbname = "Umeda_underground_0.db"
# model.dbname = "Umeda_3.db"
model.plot(footprints=True)

# fig, ax = plt.subplots(figsize=(10, 10))

# Map = np.zeros_like(model.original)
# conn = sqlite3.connect(os.path.join("data", model.paraname, model.dbname))
# c = conn.cursor()
# c.execute("SELECT seq FROM sqlite_sequence WHERE name=?", ("steps",))
# last_step_id = c.fetchone()[0]

# for frame in range(last_step_id):
#     c.execute("SELECT x, y FROM positions WHERE step_id=?", (frame + 1,))
#     data = c.fetchall()
#     for x, y in data:
#         Map[int(y), int(x)] += 1

# conn.close()

# colors = ["white", "red", "black", "green", "blue"]
# cmap = ListedColormap(colors)
# ax.imshow(model.original, cmap=cmap, vmin=0, vmax=4)

# rows, cols = np.where((Map >= 5) & (model.original == 3))
# sizes = Map[rows, cols] * 25
# ax.scatter(cols, rows, s=sizes, c="red", alpha=0.5)

# plt.tick_params(
#     labelbottom=False,
#     labelleft=False,
#     labelright=False,
#     labeltop=False,
#     bottom=False,
#     left=False,
#     right=False,
#     top=False,
# )

# # plt.show()
# plt.savefig(
#     os.path.join(
#         "output",
#         model.paraname,
#         f"{model.filename}_{model.number}_ExitUsed.png",
#     )
# )
