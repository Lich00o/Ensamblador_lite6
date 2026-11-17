"""Herramientas de línea de comandos.

  ros2 run lite6_handeye generate_board --out charuco_5x7 [--squares-x 5 ...]
  ros2 run lite6_handeye solve_offline ~/.ros/lite6_handeye/lite6_zed_calib_samples.yaml
"""

import argparse
import os

import cv2

from . import calib_io
from .charuco import BoardConfig, generate_svg, render_raster


def generate_board(argv=None):
    ap = argparse.ArgumentParser(description='Genera tablero ChArUco (SVG vectorial para láser + PNG)')
    ap.add_argument('--out', default='charuco_board', help='prefijo de salida (sin extensión)')
    ap.add_argument('--squares-x', type=int, default=5)
    ap.add_argument('--squares-y', type=int, default=7)
    ap.add_argument('--square-length', type=float, default=0.030, help='m (nominal)')
    ap.add_argument('--marker-length', type=float, default=0.022, help='m (nominal)')
    ap.add_argument('--dictionary', default='DICT_4X4_50')
    ap.add_argument('--margin', type=float, default=0.010, help='margen blanco, m')
    ap.add_argument('--png-px-per-mm', type=int, default=20)
    ap.add_argument('--invert', action='store_true',
                    help='invierte (grabar zonas BLANCAS: útil si el láser aclara el anodizado)')
    a = ap.parse_args(argv)
    cfg = BoardConfig(a.squares_x, a.squares_y, a.square_length, a.marker_length, a.dictionary)
    svg = generate_svg(cfg, a.margin)
    img = render_raster(cfg, a.png_px_per_mm, a.margin)
    if a.invert:
        svg = (svg.replace('fill="white"', 'fill="__W__"').replace('fill="black"', 'fill="white"')
               .replace('fill="__W__"', 'fill="black"'))
        img = 255 - img
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + '.svg', 'w') as f:
        f.write(svg)
    cv2.imwrite(a.out + '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    w, h = cfg.size
    print(f'{a.out}.svg  {w * 1000 + 2000 * a.margin:.1f} x {h * 1000 + 2000 * a.margin:.1f} mm '
          f'(escala 1:1, unidades en mm)')
    print(f'{a.out}.png  {img.shape[1]}x{img.shape[0]} px @ {a.png_px_per_mm} px/mm '
          f'({a.png_px_per_mm * 25.4:.0f} DPI)')
    print('IMPORTANTE: tras grabar, medir con calibrador square_length y marker_length reales '
          'y escribirlos en config/lite6_handeye.yaml (board.*).')


def solve_offline(argv=None):
    ap = argparse.ArgumentParser(description='Re-resuelve una calibración desde muestras guardadas')
    ap.add_argument('samples')
    ap.add_argument('--method', default='refine', choices=['park', 'refine'])
    ap.add_argument('--out', default=None, help='YAML de resultado (default: junto a muestras)')
    ap.add_argument('--name', default=None)
    a = ap.parse_args(argv)
    frames, G, C, T_ro = calib_io.load_samples(a.samples)
    path = calib_io.expand(a.samples)
    name = a.name or os.path.basename(path).replace('_samples.yaml', '').replace('.yaml', '')
    out = a.out or os.path.join(os.path.dirname(path), f'{name}_{a.method}.yaml')
    res, _ = calib_io.solve_and_save(out, name, frames, G, C, T_ro, method=a.method)
    m = res['metrics']
    print(f'{out}\n muestras {len(res["samples_used"])}/{len(G)}  '
          f'tablero: {m["trans_rms_mm"]:.2f} mm RMS (máx {m["trans_max_mm"]:.2f})  '
          f'{m["rot_rms_deg"]:.3f}° RMS\n {res["static_transform_publisher"]}')


def compare_gt(argv=None):
    ap = argparse.ArgumentParser(description='Compara una calibración con el ground truth de la simulación')
    ap.add_argument('result', nargs='?', default='~/.ros/lite6_handeye/lite6_zed_calib.yaml')
    ap.add_argument('--gt', default='~/.ros/lite6_handeye/sim_ground_truth.yaml')
    ap.add_argument('--max-mm', type=float, default=1.0)
    ap.add_argument('--max-deg', type=float, default=0.2)
    a = ap.parse_args(argv)
    res, gt = calib_io.load_result(a.result), calib_io.load_result(a.gt)
    err = calib_io.compare(res, gt)
    ok = True
    for k, e in err.items():
        good = e['trans_mm'] <= a.max_mm and e['rot_deg'] <= a.max_deg
        ok &= good
        print(f'{k:20s} {e["trans_mm"]:7.3f} mm  {e["rot_deg"]:6.3f}°  {"OK" if good else "FUERA"}')
    m = res.get('metrics', {})
    if m:
        print(f'consistencia: {m["trans_rms_mm"]:.2f} mm RMS, {m["rot_rms_deg"]:.3f}° RMS, '
              f'{len(res["samples_used"])}/{res["n_samples_total"]} muestras')
    print('ACEPTADO' if ok else f'RECHAZADO (umbral {a.max_mm} mm / {a.max_deg}°)')
    return 0 if ok else 1
