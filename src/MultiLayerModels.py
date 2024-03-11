import numpy as np

from .FFM import FloorFieldModel


class MainLayerModel(FloorFieldModel):
    def __init__(self, Map, SFF=None, method="L2"):
        super().__init__(Map, SFF, method)
        # メインレイヤー固有の追加初期化処理（必要に応じて）

    def process_inflow(self):
        pass

    def moving_between_layers(self, other_layer):
        pedestrian = []
        for pos in np.argwhere(other_layer.original == 3):
            if self.Map[tuple(pos)] == 1:  # mainレイヤーに歩行者がいる場合
                pedestrian.append(False)
            elif (
                other_layer.Map[tuple(pos)] == 1
            ):  # Otherレイヤーに歩行者がいる場合
                pedestrian.append(True)
            else:
                pedestrian.append(False)
        return np.argwhere(other_layer.original == 3)[pedestrian]

    def process_inflow_layer(self, pedestrian):
        for ped in pedestrian:
            if self.Map[tuple(ped)] != 2:
                # Mainレイヤーの流入口に歩行者を追加
                self.positions = np.append(self.positions, [ped], axis=0)
                self.Map[tuple(ped)] = 1  # Mainレイヤーのマップを更新


class OtherLayerModel(FloorFieldModel):
    def __init__(self, Map, SFF=None, method="L2", main_layer=None):
        super().__init__(Map, SFF, method)
        self.main_layer = main_layer  # main_layer をインスタンス変数として保存
        # 他のレイヤー固有の追加初期化処理（必要に応じて）

    def remove_pedestrians(self):
        pass

    def remove_pedestrians_layer(self, pedestrian):
        # メインレイヤーへの流入に伴う歩行者の除去
        to_remove = []
        for i, pos in enumerate(self.positions):
            if pos in pedestrian:
                to_remove.append(i)
        self.positions = np.delete(self.positions, to_remove, axis=0)
