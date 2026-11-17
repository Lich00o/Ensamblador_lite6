#!/usr/bin/env python3
"""Prueba del re-ID de ByteTrack con la cámara en vivo, sin robot ni ROS.

Lee el cuadro que deja el puente ffmpeg del host (~/.ros/lite6_handeye/.live_cam.jpg), corre
YOLO26n-pose + ByteTracker con galería (embedding OSNet-AIN) y registra los eventos de ID (nuevo, recuperado, a la
galería) y la similitud de apariencia de cada track nuevo contra la galería, para ajustar
`reid_thresh`. Video anotado en http://localhost:8090 (MJPEG).

Uso (en lite6_gpu): python3 reid_live_test.py [--thresh 0.8] [--buffer-s 3] [--reid-s 120]
Prueba: ponerse frente a la cámara, salir del cuadro > buffer-s segundos y volver por otro lado.
"""
import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import yaml

from lite6_handeye import byte_tracker, person_tracker

DATA = os.path.expanduser('~/.ros/lite6_handeye')
latest = {'jpg': None}


class Mjpeg(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
        self.end_headers()
        try:
            while True:
                j = latest['jpg']
                if j is not None:
                    self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + j + b'\r\n')
                time.sleep(0.07)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thresh', type=float, default=0.6)
    ap.add_argument('--model', default=f'{DATA}/models/osnet_ain_x1_0_msmt17.onnx')
    ap.add_argument('--buffer-s', type=float, default=3.0)
    ap.add_argument('--reid-s', type=float, default=120.0)
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--dump', default='', help='carpeta: guarda recorte + embedding por track cada 10 cuadros')
    a = ap.parse_args()

    K = np.array(yaml.safe_load(open(f'{DATA}/orbbec_intrinsics.yaml'))['camera_matrix']).reshape(3, 3)
    det = person_tracker.PersonDetector(onnx_model=f'{DATA}/models/yolo26n-pose.onnx', min_score=0.15)
    print('detector:', det.net_provider)
    fps = 20.0                                    # ritmo del bucle (se mide abajo)
    emb = byte_tracker.ReIDEmbedder(a.model)
    print('reid:', emb.provider)
    sim = byte_tracker.cosine_similarity
    bt = byte_tracker.ByteTracker(new_track_thresh=0.6, track_buffer=int(a.buffer_s * fps),
                                  reid_thresh=a.thresh, reid_buffer=int(a.reid_s * fps),
                                  reid_probation=int(fps), similarity=sim)
    srv = ThreadingHTTPServer(('0.0.0.0', a.port), Mjpeg)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f'video: http://localhost:{a.port}')

    last_mtime, prev_ids, prev_gal, t0, n = 0.0, set(), set(), time.time(), 0
    while True:
        path = f'{DATA}/.live_cam.jpg'
        try:
            m = os.path.getmtime(path)
        except OSError:
            time.sleep(0.05)
            continue
        if m == last_mtime:
            time.sleep(0.01)
            continue
        last_mtime = m
        bgr = cv2.imread(path)
        if bgr is None:                           # ffmpeg escribiendo
            continue
        dets = det.detect_all(bgr, K, 0.15)
        for d, f in zip(dets, emb(bgr, [d.bbox for d in dets])):
            d.feat = f
        gal_before = dict(bt.gallery)
        tracks = bt.update(dets)
        n += 1
        if a.dump and n % 10 == 0:
            os.makedirs(a.dump, exist_ok=True)
            H, W = bgr.shape[:2]
            for t in tracks:
                if t.visible and t.det.feat is not None:
                    x, y, w, h = (int(v) for v in t.det.bbox)
                    crop = bgr[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
                    stem = f'{a.dump}/id{t.id:03d}_{n:06d}'
                    cv2.imwrite(stem + '.jpg', crop)
                    np.save(stem + '.npy', t.det.feat)

        ids = {t.id for t in tracks}
        ts = time.strftime('%H:%M:%S')
        for old, new in bt.aliases.items():
            print(f'{ts}  RECUPERADO  ID {new} (track nuevo {old} reconocido)')
        for t in tracks:
            if t.id not in prev_ids and t.id not in bt.aliases.values() and t.feat is not None and gal_before:
                sims = {k: round(sim(t.feat, f), 3) for k, (f, _) in gal_before.items()}
                print(f'{ts}  NUEVO       ID {t.id}  similitud vs galería: {sims}')
            elif t.id not in prev_ids and t.id not in bt.aliases.values():
                print(f'{ts}  NUEVO       ID {t.id}')
        for k in set(bt.gallery) - prev_gal:
            print(f'{ts}  A GALERÍA   ID {k}')
        prev_ids, prev_gal = ids, set(bt.gallery)
        # probación: similitud en vivo de tracks nuevos (para ajustar el umbral)
        for t in tracks:
            if t.probation and t.visible and t.feat is not None and bt.gallery and n % 5 == 0:
                sims = {k: round(sim(t.feat, f), 3) for k, (f, _) in bt.gallery.items()}
                print(f'{ts}    en prueba ID {t.id}: {sims}')

        dbg = bgr.copy()
        for t in tracks:
            if not t.visible:
                continue
            x, y, w, h = (int(v) for v in t.bbox)
            c = (0, 200, 255) if t.probation else (0, 255, 0)
            cv2.rectangle(dbg, (x, y), (x + w, y + h), c, 2)
            cv2.putText(dbg, f'ID {t.id}', (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
        dt = time.time() - t0
        cv2.putText(dbg, f'{n / max(dt, 1e-3):.1f} fps  galeria: {sorted(bt.gallery)}', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        latest['jpg'] = cv2.imencode('.jpg', dbg, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        if n == 60:
            print(f'(ritmo medido: {n / dt:.1f} cuadros/s; buffers asumen {fps:.0f})')


if __name__ == '__main__':
    main()
