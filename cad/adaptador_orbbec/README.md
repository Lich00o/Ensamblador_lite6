# Adaptador Orbbec Astra Pro Plus + Vacuum Gripper Lite para el Lite 6

Este adaptador va entre la brida del Lite 6 y la ventosa (Vacuum Gripper Lite), y lleva la cámara
Orbbec Astra Pro Plus de costado. Se compone de una **cuna** impresa y una **tapa** que aprieta la
cámara con 4 tornillos M4 a insertos térmicos.

| Archivo | Contenido |
|---|---|
| `generar_soporte.py` | Generador paramétrico (FreeCAD). Todas las cotas están al inicio del archivo |
| `soporte_cuna.step/.stl` | Cuna **v3**: disco de 10 mm, hueco de cable de 33 × 17 mm |
| `soporte_tapa.step/.stl` | Tapa |
| `asm.json` | Script de `solidcraft-cli` que arma el ensamble para revisarlo |
| `anteriores/disco5mm/` | Primera versión: disco de 5 mm, hueco de 30 × 14 mm |
| `anteriores/cable14mm/` | Disco de 10 mm con el hueco de cable anterior |

## Geometría

- **Marco:** el origen está en el centro de la brida y Z+ sale del robot. Coincide con `link_eef`.
- **Disco del sándwich:** la cara z = 0 apoya en la brida y la ventosa apoya en z = 10 mm. Lleva 2 agujeros de Ø6.4 a 63 mm para los tornillos de la ventosa, que ahora deben ser 5 mm más largos que con la versión anterior.
- **Cámara:** el centro queda a y = −70 mm, con la lente RGB en x = +12.5 mm. De ahí sale el `x_prior` de la calibración, `(12.5, −70, 42) mm, yaw 180°`.
- **Hueco del cable:** 33 × 17 mm en el piso de la cuna, para que pase el filtro del cable USB, que no se puede desmontar.

## Regenerar

El generador necesita el STEP de la cámara para colocarla y verificar que no haya interferencias.
No está en este repositorio porque es un modelo de Orbbec. El archivo que se usó se llama
`Orbbec Astra Camera - Composite part 1.step`. Cópialo en esta carpeta y ejecuta:

```bash
freecadcmd -c "exec(open('generar_soporte.py').read())"
```

El script reporta la interferencia entre cuna, tapa y cámara (debe ser 0 mm³) y la masa estimada en
PETG macizo.

Para armar el ensamble de revisión (`asm.json`) también hacen falta `camara_posicionada.step`, que
genera el script, y `gripper_posicionado.step`. Este último es el STEP del Vacuum Gripper Lite de
UFactory, desplazado +10 mm en z.

## Impresión y carga

| Pieza | Laminado (Bambu Studio, 0.20 mm, 2 paredes, 15 % de relleno) | Tiempo |
|---|---|---|
| Cuna v3 | 73.4 g en PLA / 72.3 g en PETG | 1 h 54 min |
| Tapa | 11 g | 21 min |

Las dos piezas se imprimen sin soportes. La cuna va con la cara de la brida sobre la cama.

**Herramienta completa:** cuna + tapa + cámara (310 g) + ventosa (250 g) + tornillos = **0.658 kg**,
con el centro de gravedad en **(0, −40, 28) mm** en el marco de la brida. Es lo que se carga con
`set_tcp_load` (ver `docs/ARQUITECTURA_SEGUIMIENTO.md`, sección 2.4). Supera los 600 g de carga
nominal del Lite 6.

`src/lite6_handeye/sim/meshes/soporte_camara.stl` es la cuna v3 más la tapa, en metros, para Gazebo.
