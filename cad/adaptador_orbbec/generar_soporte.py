# Soporte Orbbec Astra para TCP Lite 6 + tapa con M4 a insertos.
# Ejecutar: freecadcmd -c "exec(open('generar_soporte.py').read())"
# Marco: origen = centro del flange, Z+ sale del robot. Disco z 0..BASE_T.
import FreeCAD as App, Part
from FreeCAD import Vector as V

# --- Camara (medida del STEP: estadio 165 x 30, prof. 40, frente z=0) ---
CAM_L, CAM_H, CAM_D = 164.85, 30.0, 40.0
CAM_Y = -70.0            # offset centro camara vs eje flange (igual que diseno previo)
CLR = 0.3                # holgura por lado en el bolsillo

# --- Flange (de "mounting tool.step") ---
DISC_R, BASE_T = 38.75, 5.0
DISC_EXTRA = 5.0         # disco del sandwich mas grueso solo hacia la herramienta (+Z); cara del flange en z=0
DISC_T = BASE_T + DISC_EXTRA
FLANGE_HOLE_D, FLANGE_HOLE_X = 6.4, 31.5

# --- Herramienta del robot montada sobre el disco ---
TOOL_CLEAR_R = 41.0      # radio libre sobre el disco (gripper 38.9 + 2 mm)
RIB_H = 15.0

# --- Cuna ---
WALL = 3.0
PRELOAD = 0.5            # pared queda bajo el frente de la camara -> tapa prensa la camara
CAM_BACK_Z = BASE_T      # trasera de camara apoyada en el piso
CAM_FRONT_Z = CAM_BACK_Z + CAM_D
WALL_TOP = CAM_FRONT_Z - PRELOAD
FLOOR_LEDGE = 6.0
CABLE_SLOT = (33.0, 17.0)            # salida USB por el piso (x, y); +3 mm para que pase el filtro del cable
CABLE_Y = CAM_Y - 5.0                # stub USB en y cam -7..-3
BUMP_NOTCH_W = 27.0                  # lengueta bisagra: x +-12.5, 7.5..15 mm tras el frente
BUMP_NOTCH_BOTTOM = CAM_FRONT_Z - 16.0

# --- M4 insertos ---
INS_HOLE_D, INS_DEPTH = 5.6, 9.0     # ajustar a diam. ext. del inserto - 0.3..0.4
BOSS_D = 11.0
BOSS_X = 50.0
BOSS_DY = 21.0                       # desde el centro de la camara
SCREW_CLR_D = 4.5

# --- Tapa ---
LID_T = 5.0
LIP = 4.0                            # cuanto pisa la tapa el borde del frente


def stadium(L, H, h, z0=0.0, cx=0.0, cy=0.0):
    r = H / 2.0
    box = Part.makeBox(L - H, H, h, V(cx - (L - H) / 2, cy - r, z0))
    c1 = Part.makeCylinder(r, h, V(cx - (L - H) / 2, cy, z0))
    c2 = Part.makeCylinder(r, h, V(cx + (L - H) / 2, cy, z0))
    return box.fuse([c1, c2]).removeSplitter()


bosses = [(sx * BOSS_X, CAM_Y + sy * BOSS_DY) for sx in (-1, 1) for sy in (-1, 1)]
in_L, in_H = CAM_L + 2 * CLR, CAM_H + 2 * CLR
out_L, out_H = in_L + 2 * WALL, in_H + 2 * WALL

# ---------------- Cuna ----------------
disc = Part.makeCylinder(DISC_R, DISC_T)
bridge = Part.makeBox(2 * DISC_R, abs(CAM_Y), BASE_T, V(-DISC_R, CAM_Y, 0))
tub = stadium(out_L, out_H, WALL_TOP, 0, 0, CAM_Y)
cradle = disc.fuse([bridge, tub])
for (x, y) in bosses:
    cradle = cradle.fuse(Part.makeCylinder(BOSS_D / 2, WALL_TOP, V(x, y, 0)))
# Nervio entre disco y cuna, fuera de la huella de la herramienta del robot
# (Vacuum Gripper Lite: |y| <= 38.9 sobre el disco). Nada sobre el disco salvo la herramienta.
rib_y0 = CAM_Y + out_H / 2
cradle = cradle.fuse(Part.makeBox(2 * DISC_R, (-TOOL_CLEAR_R) - rib_y0, RIB_H, V(-DISC_R, rib_y0, 0)))
cradle = cradle.removeSplitter()

pocket = stadium(in_L, in_H, CAM_D + 10, CAM_BACK_Z, 0, CAM_Y)
cradle = cradle.cut(pocket)
# salida cable USB
cradle = cradle.cut(stadium(CABLE_SLOT[0], CABLE_SLOT[1], BASE_T + 2, -1, 0, CABLE_Y))
# ventanas de alivianamiento en el piso (queda repisa de FLOOR_LEDGE)
for sx in (-1, 1):
    cradle = cradle.cut(stadium(55.0, CAM_H - 2 * FLOOR_LEDGE, BASE_T + 2, -1, sx * 47.5, CAM_Y))
# muesca lengueta bisagra (lado -Y de la camara, lejos del disco)
cradle = cradle.cut(Part.makeBox(BUMP_NOTCH_W, WALL + 6, 50,
                                 V(-BUMP_NOTCH_W / 2, CAM_Y - out_H / 2 - 4, BUMP_NOTCH_BOTTOM)))
# agujeros flange
for sx in (-1, 1):
    cradle = cradle.cut(Part.makeCylinder(FLANGE_HOLE_D / 2, DISC_T + 2, V(sx * FLANGE_HOLE_X, 0, -1)))
# agujeros insertos M4
for (x, y) in bosses:
    cradle = cradle.cut(Part.makeCylinder(INS_HOLE_D / 2, INS_DEPTH + 1, V(x, y, WALL_TOP - INS_DEPTH)))
cradle = cradle.removeSplitter()

# ---------------- Tapa ----------------
lid_z = CAM_FRONT_Z
lid = stadium(out_L, out_H, LID_T, lid_z, 0, CAM_Y)
ear_y = BOSS_DY
lid = lid.fuse(Part.makeBox(2 * BOSS_X, 2 * ear_y, LID_T, V(-BOSS_X, CAM_Y - ear_y, lid_z)))
for (x, y) in bosses:
    lid = lid.fuse(Part.makeCylinder(BOSS_D / 2, LID_T, V(x, y, lid_z)))
lid = lid.removeSplitter()
lid = lid.cut(stadium(CAM_L - 2 * LIP, CAM_H - 2 * LIP, LID_T + 2, lid_z - 1, 0, CAM_Y))
for (x, y) in bosses:
    lid = lid.cut(Part.makeCylinder(SCREW_CLR_D / 2, LID_T + 2, V(x, y, lid_z - 1)))
lid = lid.removeSplitter()

# ---------------- Camara posicionada (referencia) ----------------
cam = Part.Shape()
cam.read("Orbbec Astra Camera - Composite part 1.step")
main = cam.Solids[4].copy()          # cuerpo; sin base/soporte (Parts 1 y 4)
glasses = [cam.Solids[i].copy() for i in (1, 2, 5, 6)]
camc = Part.makeCompound([main] + glasses)
camc.translate(V(0, CAM_Y, CAM_FRONT_Z))

# ---------------- Checks ----------------
inter = cradle.common(main.translated(V(0, CAM_Y, CAM_FRONT_Z)))
print("interferencia cuna-camara mm3:", round(inter.Volume, 2))
inter2 = lid.common(main.translated(V(0, CAM_Y, CAM_FRONT_Z)))
print("interferencia tapa-camara mm3:", round(inter2.Volume, 2))
for n, s in (("cuna", cradle), ("tapa", lid)):
    print(n, "valido:", s.isValid(), "solidos:", len(s.Solids), "vol cm3:", round(s.Volume / 1000, 1),
          "masa PETG g:", round(s.Volume / 1000 * 1.27, 1), "bbox:", s.BoundBox)

cradle.exportStep("soporte_cuna.step")
lid.exportStep("soporte_tapa.step")
camc.exportStep("camara_posicionada.step")
print("OK")
