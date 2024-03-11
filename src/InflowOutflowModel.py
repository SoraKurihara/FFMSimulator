import numpy as np

from FloorFieldModel import FloorFieldModel


class InflowOutflowModel(FloorFieldModel):
    def __init__(self, map_path, method="L2", layer_type="main"):
        super().__init__(map_path, method, layer_type)
        # 追加の初期化処理（必要に応じて）

    def is_entrance_empty(self):
        # 流入口（4）の状態を確認
        entrance_positions = np.argwhere(self.original == 4)
        for pos in entrance_positions:
            if self.Map[pos[0], pos[1]] == 1:  # 歩行者がいる場合
                return False
        return True

    def move_outflow_pedestrians(self, other_layer):
        # 他のレイヤーの流出口にいる歩行者を特定し、メインレイヤーに移動
        if self.is_entrance_empty():
            exit_positions = np.argwhere(
                other_layer.original == 3
            )  # 他のレイヤーの流出口
            for pos in exit_positions:
                if other_layer.Map[pos[0], pos[1]] == 1:  # 歩行者がいる場合
                    # 対応するメインレイヤーの流入口に移動
                    self.Map[pos[0], pos[1]] = 1
                    other_layer.Map[pos[0], pos[1]] = 0

    def update(self):
        # 既存の update 処理
        super().update()

        # 流出歩行者の移動処理（他のレイヤーが必要な場合）
        # 例: self.move_outflow_pedestrians(other_layer)
