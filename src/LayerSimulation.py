from FloorFieldModel import FloorFieldModel, OtherLayerModel


def run_simulation(params):
    # メインレイヤーと他のレイヤーのインスタンスを作成
    main_layer_model = FloorFieldModel(
        params["main_map"], method="Linf", layer_type="main"
    )
    other_layer_model = OtherLayerModel(
        params["other_map"], method="Linf", layer_type="other"
    )
    other_layer_model.set_main_layer_model(main_layer_model)

    # 統合されたシミュレーションループ
    for step in range(params["steps"]):
        main_layer_model.update()
        other_layer_model.update()


# 必要に応じて他の関数やクラスを定義
