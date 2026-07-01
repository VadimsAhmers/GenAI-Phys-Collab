import mph

client = mph.start()
print("COMSOL version:", client.version)

model = client.load("models/mirrorModel.mph")  # тут же проверка совместимости версий
print("Loaded:", model.name())

print("\n=== Параметры (ищем r, h, y_cut, r_cut, частоту) ===")
for name in model.parameters():
    print(f"  {name} = {model.parameter(name)}")

print("\n=== Что можно вычислить / выгрузить ===")
# у mph есть обход дерева модели; посмотрим узлы результатов/исследований
try:
    print(
        "Studies:",
        model / "studies" if hasattr(model, "__truediv__") else model.studies(),
    )
except Exception as e:
    print("studies:", e)

import time

t0 = time.time()
model.solve()  # запускает port sweep (полный 3D FEM)
print(f"solved in {time.time() - t0:.1f} s")


def dump(group):
    try:
        nodes = list(model / group)
    except Exception as e:
        print(f"[{group}] нет группы: {e}")
        return
    print(f"[{group}] {len(nodes)}: {[n.name() for n in nodes]}")


for g in [
    "evaluations",
    "tables",
    "datasets",
    "variables",
    "physics",
    "solutions",
    "studies",
]:
    dump(g)

# формула r_RR внутри узлов Derived Values:
print("\n--- выражения в evaluations ---")
for n in model / "evaluations":
    for prop in ("expr", "descr", "table"):
        try:
            print(f"{n.name()}.{prop} = {n.property(prop)}")
        except Exception:
            pass

node = model / "evaluations" / "r_RR"
res = node.java.getReal()  # запускает вычисление, возвращает Java-массив
val = float(res[0][0])
print(f"|r_RR|^2 = {val:.4f}   objective 1-|r_RR|^2 = {1 - val:.4f}")
# Если getReal() заартачится — запасной вариант через прямое выражение:
expr = (
    "abs(1/2*(withsol('sol4',(ewfd.S11),setval(freq,freq))"
    "-withsol('sol5',(ewfd.S22),setval(freq,freq))"
    "-1i*(withsol('sol4',(ewfd.S21),setval(freq,freq))"
    "+withsol('sol5',(ewfd.S12),setval(freq,freq)))))^2"
)
print(model.evaluate(expr))
