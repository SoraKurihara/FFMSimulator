from src.FloorFieldModel import FloorFieldModel

model = FloorFieldModel(r"map/Umeda_underground.npy", method="Linf")
# model.params(N=0, inflow=3000, k_S=3, k_D=1, k_Dir=None, k_Str=None, d="Moore")
model.params(N=10000, k_S=3, k_D=1, k_Dir=None, k_Str=None, d="Moore")
model.run(steps=10000)
model.plot(footprints=True)
