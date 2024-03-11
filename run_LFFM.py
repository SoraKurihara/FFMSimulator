from tqdm import tqdm

from src.MultiLayerModels import MainLayerModel, OtherLayerModel


def run_simulation(main_map, other_map, steps):
    main_layer = MainLayerModel(main_map, SFF=True, method="Linf")
    other_layer = OtherLayerModel(
        other_map, SFF=True, method="Linf", main_layer=main_layer
    )
    main_layer.params(N=0, inflow=True, k_S=3, k_D=1, d="Moore")
    other_layer.params(N=10000, k_S=3, k_D=1, d="Moore")

    for _ in tqdm(range(steps)):
        # 各レイヤーの更新
        main_layer.update_step()
        other_layer.update_step()

        pedestrian = main_layer.moving_between_layers(other_layer)

        # 他のレイヤーからメインレイヤーへの流入処理
        main_layer.process_inflow_layer(pedestrian)

        # 他のレイヤーの更新（メインレイヤーへの流入に伴う歩行者の除去を含む）
        other_layer.remove_pedestrians_layer(pedestrian)

        if len(main_layer.positions) + len(other_layer.positions) == 0:
            break

    main_layer.plot(footprints=True)
    other_layer.plot(footprints=True)

    import os
    import sqlite3

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(10, 10))

    Map = np.zeros_like(other_layer.original)
    conn = sqlite3.connect(
        os.path.join("data", other_layer.paraname, other_layer.dbname)
    )
    c = conn.cursor()
    c.execute("SELECT seq FROM sqlite_sequence WHERE name=?", ("steps",))
    last_step_id = c.fetchone()[0]

    for frame in range(last_step_id):
        c.execute("SELECT x, y FROM positions WHERE step_id=?", (frame + 1,))
        data = c.fetchall()
        for x, y in data:
            Map[int(y), int(x)] += 1

    conn.close()

    colors = ["white", "red", "black", "green", "blue"]
    cmap = ListedColormap(colors)
    ax.imshow(other_layer.original, cmap=cmap, vmin=0, vmax=4)

    rows, cols = np.where((Map >= 5) & (other_layer.original == 3))
    sizes = Map[rows, cols] * 25
    ax.scatter(cols, rows, s=sizes, c="red", alpha=0.5)

    plt.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        bottom=False,
        left=False,
        right=False,
        top=False,
    )

    # plt.show()
    plt.savefig(
        os.path.join(
            "output",
            other_layer.paraname,
            f"{other_layer.filename}_{other_layer.number}_ExitUsed.png",
        )
    )


if __name__ == "__main__":
    run_simulation(r"map/Umeda.npy", r"map/Umeda_underground.npy", 10000)
