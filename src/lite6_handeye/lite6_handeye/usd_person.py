"""Importador y conversor de modelos humanos OpenUSD (.usd/.usda/.usdc/.usdz) para Gazebo Harmonic.

Soporta tanto personajes de **NVIDIA Omniverse / Isaac Sim (`People/Characters`, OmniPBR + UsdSkel)**
como escenarios OpenUSD estándar (`UsdPreviewSurface`):
  1. Abre escenarios `.usd`, `.usda`, `.usdc` o paquetes `.usdz` usando OpenUSD (`pxr.Usd`).
  2. Si el modelo contiene un esqueleto `UsdSkel` en pose T (como los personajes de NVIDIA Isaac Sim),
     pliega suavemente los brazos a una postura natural mediante rotación en arco (sin colapso de cuerda)
     y segmenta la malla en **13 eslabones anatómicos funcionales** (`pelvis`, `torso_link`, `head_link`,
     `arm_l_link`, `forearm_l_link`, `arm_r_link`, `forearm_r_link`, `leg_l_link`, `calf_l_link`,
     `foot_l_link`, `leg_r_link`, `calf_r_link`, `foot_r_link`) usando los pivotes óseos exactos del
     esqueleto `UsdSkel`.
  3. Extrae coordenadas UV (`primvars:st0`, `primvars:st`, `ComputeFlattened()`) y texturas PBR
     de shaders `OmniPBR` (`diffuse_texture`, `normalmap_texture`) o `UsdPreviewSurface`.
  4. Escala a estatura humana realista (`target_height`), orienta al personaje mirando hacia el robot (`-X`)
     y apoya los pies en `ground_z`.
  5. Exporta archivos `.obj` + `.mtl` y genera `person_target/model.sdf` con materiales `<pbr>`
     nativos de Gazebo Harmonic (Ogre2) y los **16 controladores articulares a 1000 Hz**.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import os
import shutil
import tempfile
import zipfile
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

LINKS = (
    'pelvis',
    'torso_link',
    'head_link',
    'arm_l_link',
    'forearm_l_link',
    'arm_r_link',
    'forearm_r_link',
    'leg_l_link',
    'calf_l_link',
    'foot_l_link',
    'leg_r_link',
    'calf_r_link',
    'foot_r_link',
)

JOINTS = (
    'px', 'py', 'pz',
    'yaw', 'pitch', 'head_yaw',
    'arm_l', 'elbow_l',
    'arm_r', 'elbow_r',
    'leg_l', 'knee_l', 'ankle_l',
    'leg_r', 'knee_r', 'ankle_r',
)

DEFAULT_PIVOTS: Dict[str, np.ndarray] = {
    'pelvis': np.array([0.0, 0.0, 0.0], dtype=float),
    'torso_link': np.array([0.0, 0.0, 0.06], dtype=float),
    'head_link': np.array([0.0, 0.0, 0.44], dtype=float),
    'arm_l_link': np.array([0.0, 0.18, 0.36], dtype=float),
    'forearm_l_link': np.array([0.0, 0.24, 0.12], dtype=float),
    'arm_r_link': np.array([0.0, -0.18, 0.36], dtype=float),
    'forearm_r_link': np.array([0.0, -0.24, 0.12], dtype=float),
    'leg_l_link': np.array([0.0, 0.09, -0.04], dtype=float),
    'calf_l_link': np.array([0.0, 0.09, -0.45], dtype=float),
    'foot_l_link': np.array([0.0, 0.08, -0.88], dtype=float),
    'leg_r_link': np.array([0.0, -0.09, -0.04], dtype=float),
    'calf_r_link': np.array([0.0, -0.09, -0.45], dtype=float),
    'foot_r_link': np.array([0.0, -0.08, -0.88], dtype=float),
}

SKEL_GROUPS = (
    'torso_link',
    'pelvis',
    'head_link',
    'skel_arm_pos_x',
    'skel_forearm_pos_x',
    'skel_arm_neg_x',
    'skel_forearm_neg_x',
    'skel_thigh_pos_x',
    'skel_calf_pos_x',
    'skel_foot_pos_x',
    'skel_thigh_neg_x',
    'skel_calf_neg_x',
    'skel_foot_neg_x',
)

ADJACENT_SKEL_PAIRS: Set[Tuple[str, str]] = {
    ('torso_link', 'head_link'),
    ('pelvis', 'torso_link'),
    ('torso_link', 'skel_arm_pos_x'),
    ('torso_link', 'skel_arm_neg_x'),
    ('skel_arm_pos_x', 'skel_forearm_pos_x'),
    ('skel_arm_neg_x', 'skel_forearm_neg_x'),
    ('pelvis', 'skel_thigh_pos_x'),
    ('pelvis', 'skel_thigh_neg_x'),
    ('skel_thigh_pos_x', 'skel_calf_pos_x'),
    ('skel_thigh_neg_x', 'skel_calf_neg_x'),
    ('skel_calf_pos_x', 'skel_foot_pos_x'),
    ('skel_calf_neg_x', 'skel_foot_neg_x'),
}


@dataclass
class MaterialSpec:
    name: str
    diffuse: Tuple[float, float, float] = (0.78, 0.78, 0.78)
    specular: Tuple[float, float, float] = (0.12, 0.12, 0.12)
    roughness: float = 0.65
    opacity: float = 1.0
    texture_path: Optional[str] = None
    normal_path: Optional[str] = None


@dataclass
class SubMesh:
    name: str
    vertices: np.ndarray                    # (N, 3) float64
    faces: np.ndarray                       # (M, 3) int32 (índices en vertices)
    uvs: Optional[np.ndarray] = None        # (K, 2) float64
    face_uvs: Optional[np.ndarray] = None   # (M, 3) int32 (índices en uvs)
    material: MaterialSpec = field(default_factory=lambda: MaterialSpec('default_mat'))
    link: Optional[str] = None
    tri_links: Optional[List[str]] = None   # clasificación primaria por triángulo desde UsdSkel
    tri_secondary_links: Optional[List[Optional[str]]] = None  # solape de costura articular


def classify_prim_link(prim_path: str) -> Optional[str]:
    """Clasifica una ruta/nombre de prim USD sin esqueleto en uno de los eslabones articulados."""
    s = prim_path.lower().replace('-', '_')
    if any(k in s for k in ('forearm_l', 'leftforearm', 'lefthand', 'hand_l', 'glove_l')):
        return 'forearm_l_link'
    if any(k in s for k in ('forearm_r', 'rightforearm', 'righthand', 'hand_r', 'glove_r')):
        return 'forearm_r_link'
    if any(k in s for k in ('arm_l', 'leftarm', 'left_arm', 'l_arm', 'leftshoulder', 'sleeve_l')):
        return 'arm_l_link'
    if any(k in s for k in ('arm_r', 'rightarm', 'right_arm', 'r_arm', 'rightshoulder', 'sleeve_r')):
        return 'arm_r_link'
    if any(k in s for k in ('foot_l', 'leftfoot', 'shoe_l', 'leftshoe', 'boot_l')):
        return 'foot_l_link'
    if any(k in s for k in ('foot_r', 'rightfoot', 'shoe_r', 'rightshoe', 'boot_r')):
        return 'foot_r_link'
    if any(k in s for k in ('calf_l', 'leftcalf', 'shin_l', 'leftshin')):
        return 'calf_l_link'
    if any(k in s for k in ('calf_r', 'rightcalf', 'shin_r', 'rightshin')):
        return 'calf_r_link'
    if any(k in s for k in ('leg_l', 'leftleg', 'left_leg', 'l_leg', 'leftupleg', 'leftthigh')):
        return 'leg_l_link'
    if any(k in s for k in ('leg_r', 'rightleg', 'right_leg', 'r_leg', 'rightupleg', 'rightthigh')):
        return 'leg_r_link'
    if any(k in s for k in ('pelvis', 'hips', 'waist', 'belt')):
        return 'pelvis'
    if any(k in s for k in ('head', 'hair', 'neck', 'face', 'eye', 'hardhat', 'helmet')):
        return 'head_link'
    if any(k in s for k in ('torso', 'chest', 'spine', 'shirt', 'jacket', 'vest')):
        return 'torso_link'
    return None


def _classify_skel_joint(joint_name: str) -> str:
    """Clasifica un hueso de UsdSkel en uno de los grupos anatómicos de extremidades y tronco.

    Importante: Se inspecciona primero la rama jerárquica de piernas (`l_thigh`/`r_thigh`,
    `l_calf`/`r_calf`, `l_foot`/`r_foot`) ANTES de buscar nombres de dedos (`pinky`, `ring`,
    `mid`, `index`), evitando que los dedos de los pies (`L_PinkyToe1`, `L_RingToe1`, etc.)
    se confundan con los dedos de las manos (`L_Pinky1`, `L_Ring1`, etc.).
    """
    path_l = joint_name.lower().replace('-', '_')
    leaf = path_l.split('/')[-1]

    # 1. Pierna +X (L_Thigh -> L_Calf -> L_Foot -> L_Toe*)
    if any(k in path_l for k in ('l_thigh', 'leftupleg', 'left_thigh', 'l_leg', 'leftleg')):
        if any(k in path_l for k in ('l_foot', 'l_toe', 'leftfoot', 'lefttoe', 'l_ankle', 'l_ball')):
            return 'skel_foot_pos_x'
        if any(k in path_l for k in ('l_calf', 'l_knee', 'l_shin', 'leftleg', 'leftcalf')):
            return 'skel_calf_pos_x'
        return 'skel_thigh_pos_x'

    # 2. Pierna -X (R_Thigh -> R_Calf -> R_Foot -> R_Toe*)
    if any(k in path_l for k in ('r_thigh', 'rightupleg', 'right_thigh', 'r_leg', 'rightleg')):
        if any(k in path_l for k in ('r_foot', 'r_toe', 'rightfoot', 'righttoe', 'r_ankle', 'r_ball')):
            return 'skel_foot_neg_x'
        if any(k in path_l for k in ('r_calf', 'r_knee', 'r_shin', 'rightleg', 'rightcalf')):
            return 'skel_calf_neg_x'
        return 'skel_thigh_neg_x'

    # Fallback para nombres de huesos de pierna sin prefijo Thigh en el path
    if any(k in path_l for k in ('l_foot', 'l_toe', 'leftfoot', 'lefttoe')):
        return 'skel_foot_pos_x'
    if any(k in path_l for k in ('r_foot', 'r_toe', 'rightfoot', 'righttoe')):
        return 'skel_foot_neg_x'
    if any(k in path_l for k in ('l_calf', 'l_knee', 'leftcalf')):
        return 'skel_calf_pos_x'
    if any(k in path_l for k in ('r_calf', 'r_knee', 'rightcalf')):
        return 'skel_calf_neg_x'

    # 3. Brazo +X (L_Upperarm -> L_Forearm -> L_Hand -> dedos)
    if any(k in path_l for k in ('l_upperarm', 'l_forearm', 'l_hand', 'leftarm', 'leftforearm', 'lefthand')):
        if any(k in path_l for k in ('l_forearm', 'l_hand', 'l_elbow', 'l_thumb', 'l_index',
                                     'l_mid', 'l_ring', 'l_pinky', 'leftforearm', 'lefthand')):
            return 'skel_forearm_pos_x'
        return 'skel_arm_pos_x'

    # 4. Brazo -X (R_Upperarm -> R_Forearm -> R_Hand -> dedos)
    if any(k in path_l for k in ('r_upperarm', 'r_forearm', 'r_hand', 'rightarm', 'rightforearm', 'righthand')):
        if any(k in path_l for k in ('r_forearm', 'r_hand', 'r_elbow', 'r_thumb', 'r_index',
                                     'r_mid', 'r_ring', 'r_pinky', 'rightforearm', 'righthand')):
            return 'skel_forearm_neg_x'
        return 'skel_arm_neg_x'

    # 5. Cabeza y cuello
    if any(k in path_l for k in ('neck', 'head', 'facial', 'jaw', 'tongue', 'teeth', 'eye')):
        return 'head_link'

    # 6. Pelvis / Cadera raíz
    if leaf in ('rl_boneroot', 'hip', 'pelvis', 'hips', 'root'):
        return 'pelvis'

    # 7. Torso / Espina dorsal / Clavículas / Costillas
    return 'torso_link'


def _triangulate(counts: List[int], indices: List[int],
                 has_face_varying_uv: bool = False
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convierte polígonos arbitrarios (triángulos, quads, n-gons) en triángulos."""
    tris: List[Tuple[int, int, int]] = []
    tri_uvs: List[Tuple[int, int, int]] = []
    idx_ptr = 0
    for c in counts:
        c_int = int(c)
        if c_int >= 3:
            v0 = int(indices[idx_ptr])
            for k in range(1, c_int - 1):
                v1 = int(indices[idx_ptr + k])
                v2 = int(indices[idx_ptr + k + 1])
                tris.append((v0, v1, v2))
                if has_face_varying_uv:
                    tri_uvs.append((idx_ptr, idx_ptr + k, idx_ptr + k + 1))
        idx_ptr += c_int
    faces_arr = np.asarray(tris, dtype=np.int32).reshape(-1, 3)
    fuv_arr = (np.asarray(tri_uvs, dtype=np.int32).reshape(-1, 3)
               if has_face_varying_uv and tri_uvs else None)
    return faces_arr, fuv_arr


def _resolve_asset_path(val, stage_dir: str) -> Optional[str]:
    if val is None:
        return None
    raw = str(getattr(val, 'resolvedPath', '') or getattr(val, 'path', '') or val)
    raw = raw.strip('@').strip()
    if not raw:
        return None
    candidates = [
        raw if os.path.isabs(raw) else os.path.normpath(os.path.join(stage_dir, raw)),
        os.path.join(stage_dir, 'textures', os.path.basename(raw)),
        os.path.join(stage_dir, 'Textures', os.path.basename(raw)),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _semantic_tint(name: str) -> Optional[Tuple[float, float, float]]:
    """Color realista para materiales OmniPBR cuya textura base de ropa es en escala de grises."""
    s = name.lower()
    if any(k in s for k in ('skin', 'organic', 'face', 'head', 'hair')):
        return (1.0, 1.0, 1.0)
    if 'retro__reflective' in s:
        return (0.90, 0.94, 0.82)
    if 'labcoat' in s:
        return (0.92, 0.94, 0.96)
    if 'scrubs' in s:
        return (0.16, 0.48, 0.68)
    if 'vest' in s:
        return (0.16, 0.42, 0.82)
    if any(k in s for k in ('shirt', 'tshirt', 'top', 'jacket')):
        return (0.22, 0.34, 0.52)
    if any(k in s for k in ('jeans', 'pants', 'trousers')):
        return (0.20, 0.27, 0.38)
    if 'hardhat' in s:
        return (0.95, 0.75, 0.10)
    if 'boot' in s:
        return (0.42, 0.28, 0.16)
    if 'glove' in s:
        return (0.34, 0.32, 0.28)
    if any(k in s for k in ('shoe', 'sneaker', 'leather')):
        return (0.25, 0.26, 0.28)
    return None


def _extract_material(prim, stage_dir: str) -> MaterialSpec:
    """Extrae color difuso y texturas desde OmniPBR, UsdPreviewSurface o displayColor."""
    from pxr import UsdGeom, UsdShade

    mat_name = prim.GetName() + '_mat'
    sem = _semantic_tint(prim.GetName())
    diffuse = sem if sem is not None else (0.78, 0.78, 0.78)
    roughness = 0.65
    tex_path = None
    norm_path = None

    gprim = UsdGeom.Gprim(prim)
    if gprim:
        dc_attr = gprim.GetDisplayColorAttr()
        if dc_attr and dc_attr.HasValue():
            val = dc_attr.Get()
            if val and len(val) > 0:
                c = val[0]
                diffuse = (float(c[0]), float(c[1]), float(c[2]))

    binding = UsdShade.MaterialBindingAPI(prim)
    mat, _ = binding.ComputeBoundMaterial()
    if mat:
        mat_name = mat.GetPrim().GetName()
        for child in mat.GetPrim().GetAllChildren():
            shader = UsdShade.Shader(child)
            if not shader:
                continue
            for tex_in_name in ('diffuse_texture', 'base_color_texture', 'albedo_texture'):
                inp = shader.GetInput(tex_in_name)
                if inp and inp.Get() is not None:
                    resolved = _resolve_asset_path(inp.Get(), stage_dir)
                    if resolved:
                        tex_path = resolved
                        break
            for norm_in_name in ('normalmap_texture', 'normal_texture'):
                inp = shader.GetInput(norm_in_name)
                if inp and inp.Get() is not None:
                    resolved = _resolve_asset_path(inp.Get(), stage_dir)
                    if resolved:
                        norm_path = resolved
                        break

            dc_in = shader.GetInput('diffuseColor')
            if dc_in:
                src_info = dc_in.GetConnectedSource()
                if src_info and src_info[0]:
                    tex_shader = UsdShade.Shader(src_info[0].GetPrim())
                    file_in = tex_shader.GetInput('file') if tex_shader else None
                    if file_in and file_in.Get():
                        resolved = _resolve_asset_path(file_in.Get(), stage_dir)
                        if resolved:
                            tex_path = resolved
                elif dc_in.Get() is not None:
                    c = dc_in.Get()
                    diffuse = (float(c[0]), float(c[1]), float(c[2]))

            r_in = shader.GetInput('roughness') or shader.GetInput('reflection_roughness_constant')
            if r_in and r_in.Get() is not None:
                roughness = float(r_in.Get())

    return MaterialSpec(
        name=mat_name,
        diffuse=diffuse,
        roughness=roughness,
        texture_path=tex_path,
        normal_path=norm_path,
    )


def _find_skeleton_info(stage, up_axis: str = 'Z',
                        fold_deg: float = 75.0) -> Optional[Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    """Busca un esqueleto UsdSkel.Skeleton y devuelve (joint_names, j_pos_bind_z, j_pos_posed_z)."""
    from pxr import UsdGeom, UsdSkel

    xc = UsdGeom.XformCache()
    for p in stage.Traverse():
        if p.IsA(UsdSkel.Skeleton):
            skel = UsdSkel.Skeleton(p)
            joints = skel.GetJointsAttr().Get()
            bind_tx = skel.GetBindTransformsAttr().Get()
            if not joints or not bind_tx:
                continue
            wmat = np.asarray(xc.GetLocalToWorldTransform(p), dtype=float)
            j_pos_bind: Dict[str, np.ndarray] = {}
            for jname, tx in zip(joints, bind_tx):
                local_p = np.append(np.asarray(tx, dtype=float)[3, :3], 1.0)
                pw = (local_p @ wmat)[:3]
                if str(up_axis).upper() == 'Y':
                    pz = np.array([pw[0], -pw[2], pw[1]], dtype=float)
                else:
                    pz = pw.astype(float)
                j_pos_bind[str(jname)] = pz

            # Calcular también las posiciones de articulaciones tras plegar la pose T de brazos
            piv_pos_x = None
            piv_neg_x = None
            for jn, pos in j_pos_bind.items():
                low = jn.lower()
                if low.endswith(('l_upperarm', 'leftarm')):
                    piv_pos_x = pos
                elif low.endswith(('r_upperarm', 'rightarm')):
                    piv_neg_x = pos

            j_pos_posed: Dict[str, np.ndarray] = {}
            fold_angle = math.radians(fold_deg)
            ca, sa = math.cos(fold_angle), math.sin(fold_angle)
            R_pos = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=float)
            R_neg = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]], dtype=float)

            for jn, pos in j_pos_bind.items():
                grp = _classify_skel_joint(jn)
                if piv_pos_x is not None and grp == 'skel_forearm_pos_x':
                    j_pos_posed[jn] = (pos - piv_pos_x) @ R_pos.T + piv_pos_x
                elif piv_neg_x is not None and grp == 'skel_forearm_neg_x':
                    j_pos_posed[jn] = (pos - piv_neg_x) @ R_neg.T + piv_neg_x
                else:
                    j_pos_posed[jn] = pos.copy()

            return [str(j) for j in joints], j_pos_bind, j_pos_posed
    return None


def _apply_skel_tpose_and_classify(
    pts_z: np.ndarray,
    faces: np.ndarray,
    prim,
    skel_info: Optional[Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]],
    fold_deg: float = 75.0,
) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[Optional[str]]]]:
    """Pliega los brazos desde pose T en arco suave y clasifica triángulos en 13 eslabones anatómicos."""
    from pxr import UsdGeom

    if skel_info is None:
        return pts_z, None, None
    joint_names, j_pos_bind, _ = skel_info
    primvars = UsdGeom.PrimvarsAPI(prim)
    pv_ji = primvars.GetPrimvar('skel:jointIndices')
    pv_jw = primvars.GetPrimvar('skel:jointWeights')
    if not pv_ji or not pv_jw or not pv_ji.HasValue() or not pv_jw.HasValue():
        return pts_z, None, None

    elem_size = int(pv_ji.GetElementSize() or 1)
    ji_raw = np.asarray(pv_ji.Get(), dtype=np.int32)
    jw_raw = np.asarray(pv_jw.Get(), dtype=float)
    n_verts = len(pts_z)
    if len(ji_raw) != n_verts * elem_size or len(jw_raw) != n_verts * elem_size:
        return pts_z, None, None

    ji = ji_raw.reshape(n_verts, elem_size)
    jw = jw_raw.reshape(n_verts, elem_size)

    joint_groups = [_classify_skel_joint(jn) for jn in joint_names]
    g_to_id = {g: i for i, g in enumerate(SKEL_GROUPS)}
    j_group_ids = np.array([g_to_id[g] for g in joint_groups], dtype=np.int32)

    vert_group_w = np.zeros((n_verts, len(SKEL_GROUPS)), dtype=float)
    for k in range(elem_size):
        valid = np.clip(ji[:, k], 0, len(joint_names) - 1)
        gids = j_group_ids[valid]
        np.add.at(vert_group_w, (np.arange(n_verts), gids), jw[:, k])

    prim_name_l = prim.GetName().lower()

    # Reglas de cohesión por prenda para evitar fisuras en calzado, pantalones, guantes o casco
    is_footwear = any(k in prim_name_l for k in ('shoe', 'boot', 'sneaker', 'tennisshoe'))
    is_pants = any(k in prim_name_l for k in ('jeans', 'pants', 'trousers'))
    is_shirt = any(k in prim_name_l for k in ('shirt', 'tshirt', 'top'))
    is_gloves = 'glove' in prim_name_l
    is_headgear = any(k in prim_name_l for k in ('hardhat', 'helmet'))
    is_torso_item = any(k in prim_name_l for k in ('vest', 'stethoscope', 'idbadge'))

    if is_pants:
        # Los pantalones terminan en el tobillo: fusionar cualquier peso de pie en la pantorrilla
        vert_group_w[:, g_to_id['skel_calf_pos_x']] += vert_group_w[:, g_to_id['skel_foot_pos_x']]
        vert_group_w[:, g_to_id['skel_foot_pos_x']] = 0.0
        vert_group_w[:, g_to_id['skel_calf_neg_x']] += vert_group_w[:, g_to_id['skel_foot_neg_x']]
        vert_group_w[:, g_to_id['skel_foot_neg_x']] = 0.0

    # Detectar pivotes de hombros en pose T (L_Upperarm en +X, R_Upperarm en -X)
    piv_pos_x = None
    piv_neg_x = None
    for jn, pos in j_pos_bind.items():
        low = jn.lower()
        if low.endswith(('l_upperarm', 'leftarm')):
            piv_pos_x = pos
        elif low.endswith(('r_upperarm', 'rightarm')):
            piv_neg_x = pos

    if is_shirt and piv_pos_x is not None and piv_neg_x is not None:
        # En mangas de camisa/camiseta (fuera del hombro en pose T), asegurar cohesión con el brazo
        sleeve_pos = pts_z[:, 0] > (piv_pos_x[0] + 0.025)
        sleeve_neg = pts_z[:, 0] < (piv_neg_x[0] - 0.025)
        vert_group_w[sleeve_pos, g_to_id['skel_arm_pos_x']] += vert_group_w[sleeve_pos, g_to_id['torso_link']]
        vert_group_w[sleeve_pos, g_to_id['torso_link']] = 0.0
        vert_group_w[sleeve_neg, g_to_id['skel_arm_neg_x']] += vert_group_w[sleeve_neg, g_to_id['torso_link']]
        vert_group_w[sleeve_neg, g_to_id['torso_link']] = 0.0

    pts_out = pts_z.copy()
    if piv_pos_x is not None and piv_neg_x is not None and not is_footwear and not is_pants and not is_headgear:
        fold_rad = math.radians(fold_deg)
        w_pos = np.clip(
            vert_group_w[:, g_to_id['skel_arm_pos_x']] + vert_group_w[:, g_to_id['skel_forearm_pos_x']],
            0.0, 1.0,
        )
        w_neg = np.clip(
            vert_group_w[:, g_to_id['skel_arm_neg_x']] + vert_group_w[:, g_to_id['skel_forearm_neg_x']],
            0.0, 1.0,
        )
        # Rotación en arco por vértice: theta_i = w_i * fold_rad (preserva volumen en el hombro)
        mask_pos = w_pos > 1e-4
        if np.any(mask_pos):
            th = w_pos[mask_pos] * fold_rad
            ca, sa = np.cos(th), np.sin(th)
            dp = pts_out[mask_pos] - piv_pos_x
            pts_out[mask_pos, 0] = piv_pos_x[0] + dp[:, 0] * ca + dp[:, 2] * sa
            pts_out[mask_pos, 2] = piv_pos_x[2] - dp[:, 0] * sa + dp[:, 2] * ca

        mask_neg = w_neg > 1e-4
        if np.any(mask_neg):
            th = w_neg[mask_neg] * fold_rad
            ca, sa = np.cos(th), np.sin(th)
            dp = pts_out[mask_neg] - piv_neg_x
            pts_out[mask_neg, 0] = piv_neg_x[0] + dp[:, 0] * ca - dp[:, 2] * sa
            pts_out[mask_neg, 2] = piv_neg_x[2] + dp[:, 0] * sa + dp[:, 2] * ca

    # Clasificación limpia 1-a-1 por triángulo (sin triángulos duplicados que sobresalgan al doblar)
    tri_centers_x = pts_z[faces].mean(axis=1)[:, 0]
    if is_footwear:
        tri_links = ['skel_foot_pos_x' if cx >= 0.0 else 'skel_foot_neg_x' for cx in tri_centers_x]
        return pts_out, tri_links, None
    if is_gloves:
        tri_links = ['skel_forearm_pos_x' if cx >= 0.0 else 'skel_forearm_neg_x' for cx in tri_centers_x]
        return pts_out, tri_links, None
    if is_headgear:
        return pts_out, ['head_link'] * len(faces), None
    if is_torso_item:
        return pts_out, ['torso_link'] * len(faces), None

    tri_w = vert_group_w[faces[:, 0]] + vert_group_w[faces[:, 1]] + vert_group_w[faces[:, 2]]
    best_idx = np.argmax(tri_w, axis=1)
    tri_links = [SKEL_GROUPS[int(i)] for i in best_idx]
    return pts_out, tri_links, None


def load_usd_submeshes(
    usd_path: str,
) -> Tuple[List[SubMesh], str, float, Optional[str], Optional[Dict[str, np.ndarray]]]:
    """Carga todas las mallas UsdGeom.Mesh y pivotes UsdSkel de un archivo .usd/.usda/.usdc/.usdz."""
    from pxr import Usd, UsdGeom

    usd_path = os.path.abspath(os.path.expanduser(usd_path))
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f'No existe el archivo USD: {usd_path}')

    tmpdir = None
    stage_file = usd_path
    if usd_path.lower().endswith('.usdz'):
        tmpdir = tempfile.mkdtemp(prefix='lite6_usdz_')
        with zipfile.ZipFile(usd_path, 'r') as zf:
            zf.extractall(tmpdir)
        candidates = [
            os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
            if f.lower().endswith(('.usd', '.usda', '.usdc'))
        ]
        if not candidates:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise ValueError(f'El paquete USDZ no contiene un archivo raíz USD: {usd_path}')
        stage_file = candidates[0]

    stage = Usd.Stage.Open(stage_file)
    if stage is None:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f'No se pudo abrir el escenario USD: {stage_file}')

    up_axis = str(UsdGeom.GetStageUpAxis(stage) or 'Z')
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    stage_dir = os.path.dirname(stage_file)
    skel_info = _find_skeleton_info(stage, up_axis=up_axis)

    submeshes: List[SubMesh] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get()
        cnt_attr = mesh.GetFaceVertexCountsAttr().Get()
        idx_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts_attr or not cnt_attr or not idx_attr:
            continue

        pts = np.asarray(pts_attr, dtype=float).reshape(-1, 3)
        mat4 = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=float)
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=float)])
        pts_w = (pts_h @ mat4)[:, :3]

        if up_axis.upper() == 'Y':
            pts_z = np.column_stack([pts_w[:, 0], -pts_w[:, 2], pts_w[:, 1]])
        else:
            pts_z = pts_w

        uvs_arr = None
        has_fv_uv = False
        primvars = UsdGeom.PrimvarsAPI(prim)
        for uv_name in ('st0', 'st', 'uv', 'map1'):
            pv = primvars.GetPrimvar(uv_name)
            if pv and pv.HasValue():
                flat_uv = pv.ComputeFlattened()
                raw_uv = np.asarray(flat_uv if flat_uv is not None else pv.Get(),
                                    dtype=float).reshape(-1, 2)
                interp = pv.GetInterpolation()
                if interp == UsdGeom.Tokens.faceVarying:
                    uvs_arr = raw_uv
                    has_fv_uv = True
                elif interp == UsdGeom.Tokens.vertex and len(raw_uv) == len(pts_z):
                    uvs_arr = raw_uv
                break

        faces, face_uvs = _triangulate(list(cnt_attr), list(idx_attr), has_fv_uv)
        if len(faces) == 0:
            continue
        if uvs_arr is not None and face_uvs is None and len(uvs_arr) == len(pts_z):
            face_uvs = faces.copy()

        pts_posed, tri_links, tri_sec = _apply_skel_tpose_and_classify(pts_z, faces, prim, skel_info)
        mat_spec = _extract_material(prim, stage_dir)
        link_hint = classify_prim_link(str(prim.GetPath()))
        submeshes.append(SubMesh(
            name=prim.GetName(),
            vertices=pts_posed,
            faces=faces,
            uvs=uvs_arr,
            face_uvs=face_uvs,
            material=mat_spec,
            link=link_hint,
            tri_links=tri_links,
            tri_secondary_links=tri_sec,
        ))

    if not submeshes:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise ValueError(f'No se encontraron mallas UsdGeom.Mesh en {usd_path}')

    j_pos_posed = skel_info[2] if skel_info is not None else None
    return submeshes, up_axis, mpu, tmpdir, j_pos_posed


def _extract_pivots_from_skel(
    j_pos_norm: Optional[Dict[str, np.ndarray]],
    rotated_z: bool,
) -> Dict[str, np.ndarray]:
    """Extrae los pivotes articulares exactos en el marco del modelo Gazebo desde UsdSkel."""
    pivots = {k: v.copy() for k, v in DEFAULT_PIVOTS.items()}
    if not j_pos_norm:
        return pivots

    def find_joint(suffixes: Tuple[str, ...]) -> Optional[np.ndarray]:
        for jn, pos in j_pos_norm.items():
            low = jn.lower()
            if low.endswith(suffixes):
                return pos.copy()
        return None

    # En pose rotada (-90° en Z: (x, y) -> (y, -x)), R_* (-X original) queda en +Y (izquierda Gazebo)
    # y L_* (+X original) queda en -Y (derecha Gazebo).
    l_pref = 'r_' if rotated_z else 'l_'
    r_pref = 'l_' if rotated_z else 'r_'

    mapping = {
        'torso_link': ('/waist', 'waist', '/spine01'),
        'head_link': ('/necktwist01', '/neck', '/head'),
        'arm_l_link': (f'/{l_pref}upperarm', f'{l_pref}upperarm'),
        'forearm_l_link': (f'/{l_pref}forearm', f'{l_pref}forearm'),
        'arm_r_link': (f'/{r_pref}upperarm', f'{r_pref}upperarm'),
        'forearm_r_link': (f'/{r_pref}forearm', f'{r_pref}forearm'),
        'leg_l_link': (f'/{l_pref}thigh', f'{l_pref}thigh'),
        'calf_l_link': (f'/{l_pref}calf', f'{l_pref}calf'),
        'foot_l_link': (f'/{l_pref}foot', f'{l_pref}foot'),
        'leg_r_link': (f'/{r_pref}thigh', f'{r_pref}thigh'),
        'calf_r_link': (f'/{r_pref}calf', f'{r_pref}calf'),
        'foot_r_link': (f'/{r_pref}foot', f'{r_pref}foot'),
    }
    for link_name, suffixes in mapping.items():
        found = find_joint(suffixes)
        if found is not None:
            pivots[link_name] = found
    pivots['pelvis'] = np.array([0.0, 0.0, 0.0], dtype=float)
    return pivots


def _normalize_and_orient(
    submeshes: List[SubMesh],
    j_pos_posed: Optional[Dict[str, np.ndarray]] = None,
    target_height: float = 1.58,
    ground_z: float = -0.95,
    auto_face_neg_x: bool = True,
) -> Tuple[bool, Dict[str, np.ndarray]]:
    """Escala a `target_height`, orienta mirando a `-X`, apoya pies en `ground_z` y transforma pivotes."""
    all_pts = np.vstack([m.vertices for m in submeshes])
    bmin = all_pts.min(axis=0)
    bmax = all_pts.max(axis=0)
    ext = bmax - bmin

    rotated_z = False
    if auto_face_neg_x and ext[0] > 1.18 * ext[1]:
        rotated_z = True
        for m in submeshes:
            v = m.vertices
            m.vertices = np.column_stack([v[:, 1], -v[:, 0], v[:, 2]])
        if j_pos_posed:
            for k, p in list(j_pos_posed.items()):
                j_pos_posed[k] = np.array([p[1], -p[0], p[2]], dtype=float)
        all_pts = np.vstack([m.vertices for m in submeshes])
        bmin = all_pts.min(axis=0)
        bmax = all_pts.max(axis=0)
        ext = bmax - bmin

    curr_h = float(max(ext[2], 1e-4))
    scale = (float(target_height) / curr_h) if target_height > 0.0 else 1.0
    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    z_min = bmin[2]

    for m in submeshes:
        v = m.vertices.copy()
        v[:, 0] = (v[:, 0] - cx) * scale
        v[:, 1] = (v[:, 1] - cy) * scale
        v[:, 2] = (v[:, 2] - z_min) * scale + float(ground_z)
        m.vertices = v

    if j_pos_posed:
        for k, p in list(j_pos_posed.items()):
            j_pos_posed[k] = np.array([
                (p[0] - cx) * scale,
                (p[1] - cy) * scale,
                (p[2] - z_min) * scale + float(ground_z),
            ], dtype=float)

    pivots = _extract_pivots_from_skel(j_pos_posed, rotated_z=rotated_z)
    return rotated_z, pivots


def _split_by_tri_links(submeshes: List[SubMesh], rotated_z: bool) -> List[SubMesh]:
    """Divide cada SubMesh con `tri_links` (de UsdSkel) en sub-mallas por los 13 eslabones de Gazebo."""
    remap = {
        'torso_link': 'torso_link',
        'pelvis': 'pelvis',
        'head_link': 'head_link',
        'skel_arm_pos_x': 'arm_r_link' if rotated_z else 'arm_l_link',
        'skel_forearm_pos_x': 'forearm_r_link' if rotated_z else 'forearm_l_link',
        'skel_arm_neg_x': 'arm_l_link' if rotated_z else 'arm_r_link',
        'skel_forearm_neg_x': 'forearm_l_link' if rotated_z else 'forearm_r_link',
        'skel_thigh_pos_x': 'leg_r_link' if rotated_z else 'leg_l_link',
        'skel_calf_pos_x': 'calf_r_link' if rotated_z else 'calf_l_link',
        'skel_foot_pos_x': 'foot_r_link' if rotated_z else 'foot_l_link',
        'skel_thigh_neg_x': 'leg_l_link' if rotated_z else 'leg_r_link',
        'skel_calf_neg_x': 'calf_l_link' if rotated_z else 'calf_r_link',
        'skel_foot_neg_x': 'foot_l_link' if rotated_z else 'foot_r_link',
    }
    out: List[SubMesh] = []
    for sm in submeshes:
        if not sm.tri_links:
            out.append(sm)
            continue
        buckets: Dict[str, List[int]] = {k: [] for k in LINKS}
        for tidx, tl in enumerate(sm.tri_links):
            lk = remap.get(tl, 'torso_link')
            buckets[lk].append(tidx)
            if sm.tri_secondary_links and sm.tri_secondary_links[tidx] is not None:
                sec_lk = remap.get(sm.tri_secondary_links[tidx], '')
                if sec_lk in buckets and sec_lk != lk:
                    buckets[sec_lk].append(tidx)

        for lk, tids in buckets.items():
            if not tids:
                continue
            sel = np.asarray(tids, dtype=np.int32)
            out.append(SubMesh(
                name=f'{sm.name}_{lk}',
                vertices=sm.vertices,
                faces=sm.faces[sel],
                uvs=sm.uvs,
                face_uvs=sm.face_uvs[sel] if sm.face_uvs is not None else None,
                material=sm.material,
                link=lk,
            ))
    return out


def _segment_single_mesh(m: SubMesh, ground_z: float = -0.95,
                         height: float = 1.58) -> List[SubMesh]:
    """Divide una malla humana sin esqueleto en los 13 eslabones articulados por regiones anatómicas."""
    v = m.vertices
    f = m.faces
    tri_centers = v[f].mean(axis=1)
    u_z = (tri_centers[:, 2] - ground_z) / max(height, 0.5)
    y_c = tri_centers[:, 1]
    half_shoulder = 0.125 * height

    buckets: Dict[str, List[int]] = {k: [] for k in LINKS}
    for idx in range(len(f)):
        uz, yc = float(u_z[idx]), float(y_c[idx])
        if uz < 0.08:
            buckets['foot_l_link' if yc >= 0.0 else 'foot_r_link'].append(idx)
        elif uz < 0.27:
            buckets['calf_l_link' if yc >= 0.0 else 'calf_r_link'].append(idx)
        elif uz < 0.46:
            buckets['leg_l_link' if yc >= 0.0 else 'leg_r_link'].append(idx)
        elif uz < 0.52 and abs(yc) <= half_shoulder:
            buckets['pelvis'].append(idx)
        elif 0.50 <= uz <= 0.68 and abs(yc) > half_shoulder:
            buckets['forearm_l_link' if yc > 0.0 else 'forearm_r_link'].append(idx)
        elif 0.68 < uz <= 0.86 and abs(yc) > half_shoulder:
            buckets['arm_l_link' if yc > 0.0 else 'arm_r_link'].append(idx)
        elif uz > 0.86:
            buckets['head_link'].append(idx)
        else:
            buckets['torso_link'].append(idx)

    out: List[SubMesh] = []
    for link_name, tri_ids in buckets.items():
        if not tri_ids:
            continue
        sel = np.asarray(tri_ids, dtype=np.int32)
        out.append(SubMesh(
            name=f'{m.name}_{link_name}',
            vertices=m.vertices,
            faces=f[sel],
            uvs=m.uvs,
            face_uvs=m.face_uvs[sel] if m.face_uvs is not None else None,
            material=m.material,
            link=link_name,
        ))
    return out


def _compact_submesh(sm: SubMesh) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Unifica vértices (v, vt, vn) 1-a-1 y calcula normales suaves para Ogre2 en Gazebo Harmonic."""
    v = sm.vertices
    f = sm.faces

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn_orig = np.zeros_like(v)
    np.add.at(vn_orig, f[:, 0], fn)
    np.add.at(vn_orig, f[:, 1], fn)
    np.add.at(vn_orig, f[:, 2], fn)
    norms = np.linalg.norm(vn_orig, axis=1, keepdims=True)
    vn_orig = np.where(norms > 1e-9, vn_orig / np.maximum(norms, 1e-9),
                       np.array([0.0, 0.0, 1.0]))

    if sm.uvs is not None and sm.face_uvs is not None and len(sm.face_uvs) == len(f):
        pairs = np.column_stack([f.ravel(), sm.face_uvs.ravel()])
        uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
        out_v = v[uniq_pairs[:, 0]]
        out_vn = vn_orig[uniq_pairs[:, 0]]
        out_uv = sm.uvs[uniq_pairs[:, 1]]
        out_f = inv.reshape(-1, 3).astype(np.int32)
        return out_v, out_vn, out_uv, out_f

    used_v, inv_v = np.unique(f.ravel(), return_inverse=True)
    out_v = v[used_v]
    out_vn = vn_orig[used_v]
    out_f = inv_v.reshape(-1, 3).astype(np.int32)
    return out_v, out_vn, None, out_f


def _prepare_tinted_texture(tex_src: str, tint_rgb: Tuple[float, float, float],
                            meshes_dir: str, mat_tag: str) -> str:
    """Prepara una textura optimizada (1024x1024) modulada por el color del material."""
    import cv2

    base_name = os.path.splitext(os.path.basename(tex_src))[0]
    r, g, b = tint_rgb
    out_name = f'tex_{base_name}_{mat_tag}_{int(r*100)}_{int(g*100)}_{int(b*100)}.png'
    out_path = os.path.join(meshes_dir, out_name)
    if os.path.exists(out_path):
        return out_name

    img = cv2.imread(tex_src, cv2.IMREAD_COLOR)
    if img is None:
        shutil.copy2(tex_src, out_path)
        return out_name
    if img.shape[0] > 1024 or img.shape[1] > 1024:
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

    if abs(r - 1.0) > 0.05 or abs(g - 1.0) > 0.05 or abs(b - 1.0) > 0.05:
        scale_bgr = np.array([b, g, r], dtype=np.float32).reshape(1, 1, 3)
        img = np.clip(img.astype(np.float32) * scale_bgr * 1.15, 0, 255).astype(np.uint8)

    cv2.imwrite(out_path, img)
    return out_name


def _write_obj_and_mtl(
    submeshes: List[SubMesh],
    meshes_dir: str,
    pivots: Dict[str, np.ndarray],
) -> Dict[str, List[Tuple[str, MaterialSpec, Optional[str], Optional[str]]]]:
    """Escribe archivos .obj con normales (vn) y UVs (vt) 1-a-1 en el marco local de su eslabón."""
    os.makedirs(meshes_dir, exist_ok=True)
    for old in os.listdir(meshes_dir):
        if old.startswith('usd_') and old.endswith(('.obj', '.mtl')):
            os.remove(os.path.join(meshes_dir, old))

    mtl_path = os.path.join(meshes_dir, 'usd_person.mtl')
    mtl_lines = ['# Materiales exportados desde OpenUSD para Gazebo Harmonic\n']
    link_visuals: Dict[str, List[Tuple[str, MaterialSpec, Optional[str], Optional[str]]]] = {
        k: [] for k in LINKS
    }

    for idx, raw_sm in enumerate(submeshes):
        out_v, out_vn, out_uv, out_f = _compact_submesh(raw_sm)
        link = raw_sm.link if raw_sm.link in pivots else 'torso_link'
        pivot = pivots[link]
        local_v = out_v - pivot

        mat = raw_sm.material
        mat_id = f'{mat.name}_{idx}'
        r, g, b = mat.diffuse
        sr, sg, sb = mat.specular
        mtl_lines.append(f'newmtl {mat_id}\n')
        mtl_lines.append(f'Ka {0.45 * r:.4f} {0.45 * g:.4f} {0.45 * b:.4f}\n')
        mtl_lines.append(f'Kd {r:.4f} {g:.4f} {b:.4f}\n')
        mtl_lines.append(f'Ks {sr:.4f} {sg:.4f} {sb:.4f}\n')
        mtl_lines.append(f'd {mat.opacity:.4f}\n')

        tex_rel = None
        if mat.texture_path and os.path.exists(mat.texture_path):
            tex_rel = _prepare_tinted_texture(mat.texture_path, mat.diffuse, meshes_dir, mat.name)
            mtl_lines.append(f'map_Kd {tex_rel}\n')
        mtl_lines.append('\n')

        obj_filename = f'usd_{link}_{idx}.obj'
        obj_path = os.path.join(meshes_dir, obj_filename)
        lines = [
            '# Generado por lite6_handeye.usd_person\n',
            'mtllib usd_person.mtl\n',
            f'o {raw_sm.name}\n',
            f'usemtl {mat_id}\n',
        ]
        for vx, vy, vz in local_v:
            lines.append(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
        if out_uv is not None:
            for u, v_tc in out_uv:
                lines.append(f'vt {u:.6f} {v_tc:.6f}\n')
        for nx, ny, nz in out_vn:
            lines.append(f'vn {nx:.6f} {ny:.6f} {nz:.6f}\n')

        if out_uv is not None:
            for i0, i1, i2 in out_f:
                a, b_i, c = i0 + 1, i1 + 1, i2 + 1
                lines.append(f'f {a}/{a}/{a} {b_i}/{b_i}/{b_i} {c}/{c}/{c}\n')
        else:
            for i0, i1, i2 in out_f:
                a, b_i, c = i0 + 1, i1 + 1, i2 + 1
                lines.append(f'f {a}//{a} {b_i}//{b_i} {c}//{c}\n')

        with open(obj_path, 'w') as f_obj:
            f_obj.writelines(lines)

        link_visuals[link].append((obj_filename, mat, tex_rel, None))

    with open(mtl_path, 'w') as f_mtl:
        f_mtl.writelines(mtl_lines)

    return link_visuals


def _render_sdf(
    link_visuals: Dict[str, List[Tuple[str, MaterialSpec, Optional[str], Optional[str]]]],
    pivots: Dict[str, np.ndarray],
    source_usd: str,
) -> str:
    """Genera el contenido de `person_target/model.sdf` con 13 eslabones y 16 GDL a 1000 Hz."""
    def visuals_xml(link: str) -> str:
        items = link_visuals.get(link, [])
        if not items:
            return ''
        blocks = []
        for k, (obj_file, mat, tex_rel, norm_rel) in enumerate(items):
            r, g, b = mat.diffuse
            sr, sg, sb = mat.specular
            if tex_rel:
                mat_block = f"""
        <material>
          <diffuse>1.0 1.0 1.0 1</diffuse>
          <specular>{sr:.4f} {sg:.4f} {sb:.4f} 1</specular>
          <pbr>
            <metal>
              <albedo_map>model://person_target/meshes/{tex_rel}</albedo_map>
              <roughness>{mat.roughness:.3f}</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>"""
            else:
                mat_block = f"""
        <material>
          <ambient>{0.55 * r:.4f} {0.55 * g:.4f} {0.55 * b:.4f} 1</ambient>
          <diffuse>{r:.4f} {g:.4f} {b:.4f} 1</diffuse>
          <specular>{sr:.4f} {sg:.4f} {sb:.4f} 1</specular>
        </material>"""
            blocks.append(f"""      <visual name="{link}_usd_{k}">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>model://person_target/meshes/{obj_file}</uri>
          </mesh>
        </geometry>{mat_block}
      </visual>""")
        return '\n'.join(blocks)

    def pose_str(lk: str) -> str:
        p = pivots[lk]
        return f'{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 0 0 0'

    controllers = []
    for j in JOINTS:
        vmax = '4.0' if j == 'pz' else ('3.0' if j in ('px', 'py') else '10.0')
        controllers.append(f"""    <plugin filename="gz-sim-joint-position-controller-system"
            name="gz::sim::systems::JointPositionController">
      <joint_name>{j}</joint_name>
      <topic>/person_target/{j}</topic>
      <use_velocity_commands>true</use_velocity_commands>
      <cmd_max>{vmax}</cmd_max>
      <cmd_min>-{vmax}</cmd_min>
    </plugin>""")
    controllers_xml = '\n'.join(controllers)

    return f"""<?xml version="1.0"?>
<!-- Generado automáticamente desde USD: {os.path.basename(source_usd)}
     Rig humanoide cinemático completo (13 eslabones, 16 GDL a 1000 Hz):
     - Base 4-DOF: px, py, pz, yaw (pelvis)
     - Tronco y cabeza: pitch (torso_link), head_yaw (head_link)
     - Brazos (hombro + codo): arm_l, elbow_l, arm_r, elbow_r
     - Piernas (cadera + rodilla + tobillo): leg_l, knee_l, ankle_l, leg_r, knee_r, ankle_r -->
<sdf version="1.9">
  <model name="person_target">
    <static>false</static>

    <joint name="world_fixed" type="fixed">
      <parent>world</parent>
      <child>anchor</child>
    </joint>

    <link name="anchor">
      <gravity>false</gravity>
      <inertial><mass>1.0</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
    </link>

    <link name="stage_x">
      <gravity>false</gravity>
      <inertial><mass>0.5</mass><inertia><ixx>0.005</ixx><iyy>0.005</iyy><izz>0.005</izz></inertia></inertial>
    </link>
    <joint name="px" type="prismatic">
      <parent>anchor</parent>
      <child>stage_x</child>
      <axis>
        <xyz>1 0 0</xyz>
        <limit><lower>-0.50</lower><upper>0.50</upper><effort>500</effort><velocity>3.0</velocity></limit>
      </axis>
    </joint>

    <link name="stage_y">
      <gravity>false</gravity>
      <inertial><mass>0.5</mass><inertia><ixx>0.005</ixx><iyy>0.005</iyy><izz>0.005</izz></inertia></inertial>
    </link>
    <joint name="py" type="prismatic">
      <parent>stage_x</parent>
      <child>stage_y</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-0.75</lower><upper>0.75</upper><effort>500</effort><velocity>3.0</velocity></limit>
      </axis>
    </joint>

    <link name="stage_z">
      <gravity>false</gravity>
      <inertial><mass>0.5</mass><inertia><ixx>0.005</ixx><iyy>0.005</iyy><izz>0.005</izz></inertia></inertial>
    </link>
    <joint name="pz" type="prismatic">
      <parent>stage_y</parent>
      <child>stage_z</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit><lower>-0.45</lower><upper>0.50</upper><effort>500</effort><velocity>4.0</velocity></limit>
      </axis>
    </joint>

    <link name="pelvis">
      <gravity>false</gravity>
      <pose>{pose_str('pelvis')}</pose>
      <inertial><mass>1.0</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
{visuals_xml('pelvis')}
    </link>
    <joint name="yaw" type="revolute">
      <parent>stage_z</parent>
      <child>pelvis</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit><lower>-1.57</lower><upper>1.57</upper><effort>200</effort><velocity>5.0</velocity></limit>
      </axis>
    </joint>

    <link name="torso_link">
      <gravity>false</gravity>
      <pose>{pose_str('torso_link')}</pose>
      <inertial><mass>2.0</mass><inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.02</izz></inertia></inertial>
{visuals_xml('torso_link')}
    </link>
    <joint name="pitch" type="revolute">
      <parent>pelvis</parent>
      <child>torso_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-0.65</lower><upper>0.65</upper><effort>200</effort><velocity>6.0</velocity></limit>
      </axis>
    </joint>

    <link name="head_link">
      <gravity>false</gravity>
      <pose>{pose_str('head_link')}</pose>
      <inertial><mass>0.4</mass><inertia><ixx>0.003</ixx><iyy>0.003</iyy><izz>0.003</izz></inertia></inertial>
{visuals_xml('head_link')}
    </link>
    <joint name="head_yaw" type="revolute">
      <parent>torso_link</parent>
      <child>head_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit><lower>-1.3</lower><upper>1.3</upper><effort>100</effort><velocity>8.0</velocity></limit>
      </axis>
    </joint>

    <link name="arm_l_link">
      <gravity>false</gravity>
      <pose>{pose_str('arm_l_link')}</pose>
      <inertial><mass>0.25</mass><inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.002</izz></inertia></inertial>
{visuals_xml('arm_l_link')}
    </link>
    <joint name="arm_l" type="revolute">
      <parent>torso_link</parent>
      <child>arm_l_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-1.8</lower><upper>2.9</upper><effort>100</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

    <link name="forearm_l_link">
      <gravity>false</gravity>
      <pose>{pose_str('forearm_l_link')}</pose>
      <inertial><mass>0.20</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>
{visuals_xml('forearm_l_link')}
    </link>
    <joint name="elbow_l" type="revolute">
      <parent>arm_l_link</parent>
      <child>forearm_l_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-0.3</lower><upper>2.4</upper><effort>100</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

    <link name="arm_r_link">
      <gravity>false</gravity>
      <pose>{pose_str('arm_r_link')}</pose>
      <inertial><mass>0.25</mass><inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.002</izz></inertia></inertial>
{visuals_xml('arm_r_link')}
    </link>
    <joint name="arm_r" type="revolute">
      <parent>torso_link</parent>
      <child>arm_r_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-1.8</lower><upper>2.9</upper><effort>100</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

    <link name="forearm_r_link">
      <gravity>false</gravity>
      <pose>{pose_str('forearm_r_link')}</pose>
      <inertial><mass>0.20</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>
{visuals_xml('forearm_r_link')}
    </link>
    <joint name="elbow_r" type="revolute">
      <parent>arm_r_link</parent>
      <child>forearm_r_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-0.3</lower><upper>2.4</upper><effort>100</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

    <link name="leg_l_link">
      <gravity>false</gravity>
      <pose>{pose_str('leg_l_link')}</pose>
      <inertial><mass>0.4</mass><inertia><ixx>0.003</ixx><iyy>0.003</iyy><izz>0.003</izz></inertia></inertial>
{visuals_xml('leg_l_link')}
    </link>
    <joint name="leg_l" type="revolute">
      <parent>pelvis</parent>
      <child>leg_l_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-1.6</lower><upper>1.8</upper><effort>200</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

    <link name="calf_l_link">
      <gravity>false</gravity>
      <pose>{pose_str('calf_l_link')}</pose>
      <inertial><mass>0.3</mass><inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.002</izz></inertia></inertial>
{visuals_xml('calf_l_link')}
    </link>
    <joint name="knee_l" type="revolute">
      <parent>leg_l_link</parent>
      <child>calf_l_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-2.3</lower><upper>0.2</upper><effort>200</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

    <link name="foot_l_link">
      <gravity>false</gravity>
      <pose>{pose_str('foot_l_link')}</pose>
      <inertial><mass>0.2</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>
{visuals_xml('foot_l_link')}
    </link>
    <joint name="ankle_l" type="revolute">
      <parent>calf_l_link</parent>
      <child>foot_l_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-1.2</lower><upper>1.2</upper><effort>150</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

    <link name="leg_r_link">
      <gravity>false</gravity>
      <pose>{pose_str('leg_r_link')}</pose>
      <inertial><mass>0.4</mass><inertia><ixx>0.003</ixx><iyy>0.003</iyy><izz>0.003</izz></inertia></inertial>
{visuals_xml('leg_r_link')}
    </link>
    <joint name="leg_r" type="revolute">
      <parent>pelvis</parent>
      <child>leg_r_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-1.6</lower><upper>1.8</upper><effort>200</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

    <link name="calf_r_link">
      <gravity>false</gravity>
      <pose>{pose_str('calf_r_link')}</pose>
      <inertial><mass>0.3</mass><inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.002</izz></inertia></inertial>
{visuals_xml('calf_r_link')}
    </link>
    <joint name="knee_r" type="revolute">
      <parent>leg_r_link</parent>
      <child>calf_r_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-2.3</lower><upper>0.2</upper><effort>200</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

    <link name="foot_r_link">
      <gravity>false</gravity>
      <pose>{pose_str('foot_r_link')}</pose>
      <inertial><mass>0.2</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>
{visuals_xml('foot_r_link')}
    </link>
    <joint name="ankle_r" type="revolute">
      <parent>calf_r_link</parent>
      <child>foot_r_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-1.2</lower><upper>1.2</upper><effort>150</effort><velocity>10.0</velocity></limit>
      </axis>
    </joint>

{controllers_xml}
  </model>
</sdf>
"""


def import_usd_to_person_target(
    usd_path: str,
    model_dir: str,
    target_height: float = 1.58,
    ground_z: float = -0.95,
    split_single_mesh: bool = True,
) -> dict:
    """Importa un archivo USD (.usd/.usda/.usdc/.usdz) y actualiza `person_target`."""
    submeshes, up_axis, mpu, tmpdir, j_pos_posed = load_usd_submeshes(usd_path)
    try:
        rotated_z, pivots = _normalize_and_orient(
            submeshes,
            j_pos_posed=j_pos_posed,
            target_height=target_height,
            ground_z=ground_z,
        )

        if any(sm.tri_links is not None for sm in submeshes):
            submeshes = _split_by_tri_links(submeshes, rotated_z=rotated_z)
        elif split_single_mesh and all(sm.link is None for sm in submeshes):
            segmented: List[SubMesh] = []
            for sm in submeshes:
                segmented.extend(_segment_single_mesh(sm, ground_z=ground_z, height=target_height))
            submeshes = segmented
        else:
            for sm in submeshes:
                if sm.link is None:
                    sm.link = 'torso_link'

        meshes_dir = os.path.join(model_dir, 'meshes')
        link_visuals = _write_obj_and_mtl(submeshes, meshes_dir, pivots)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    sdf_xml = _render_sdf(link_visuals, pivots, usd_path)
    sdf_path = os.path.join(model_dir, 'model.sdf')
    with open(sdf_path, 'w') as f_sdf:
        f_sdf.write(sdf_xml)

    total_verts = sum(len(sm.vertices) for sm in submeshes)
    total_tris = sum(len(sm.faces) for sm in submeshes)
    return {
        'usd_path': usd_path,
        'sdf_path': sdf_path,
        'up_axis': up_axis,
        'meters_per_unit': mpu,
        'submeshes': len(submeshes),
        'vertices': total_verts,
        'triangles': total_tris,
        'pivots': {k: p.round(4).tolist() for k, p in pivots.items()},
    }


def _ellipsoid_mesh(center: Tuple[float, float, float],
                    radii: Tuple[float, float, float],
                    n_lat: int = 14, n_lon: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Genera una malla triangular suave de un elipsoide."""
    cx, cy, cz = center
    rx, ry, rz = radii
    verts = [[cx, cy, cz + rz]]
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(n_lon):
            phi = 2.0 * math.pi * j / n_lon
            verts.append([
                cx + rx * st * math.cos(phi),
                cy + ry * st * math.sin(phi),
                cz + rz * ct,
            ])
    verts.append([cx, cy, cz - rz])
    south_idx = len(verts) - 1

    faces = []
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + jn])
    for i in range(n_lat - 2):
        row0 = 1 + i * n_lon
        row1 = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces.append([row0 + j, row1 + j, row1 + jn])
            faces.append([row0 + j, row1 + jn, row0 + jn])
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([last_row + j, south_idx, last_row + jn])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int32)


def _tapered_limb_mesh(p0: Tuple[float, float, float],
                       p1: Tuple[float, float, float],
                       r0: Tuple[float, float],
                       r1: Tuple[float, float],
                       n_rings: int = 8, n_seg: int = 18) -> Tuple[np.ndarray, np.ndarray]:
    """Genera un segmento anatómico ahusado con tapas suaves."""
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    rx0, ry0 = r0
    rx1, ry1 = r1
    verts = [[x0, y0, z0]]
    for i in range(n_rings):
        u = i / max(n_rings - 1, 1)
        bulge = 1.0 + 0.06 * math.sin(math.pi * u)
        cx = x0 + u * (x1 - x0)
        cy = y0 + u * (y1 - y0)
        cz = z0 + u * (z1 - z0)
        rx = (rx0 + u * (rx1 - rx0)) * bulge
        ry = (ry0 + u * (ry1 - ry0)) * bulge
        for j in range(n_seg):
            phi = 2.0 * math.pi * j / n_seg
            verts.append([cx + rx * math.cos(phi), cy + ry * math.sin(phi), cz])
    verts.append([x1, y1, z1])
    end_idx = len(verts) - 1

    faces = []
    for j in range(n_seg):
        jn = (j + 1) % n_seg
        faces.append([0, 1 + jn, 1 + j])
    for i in range(n_rings - 1):
        r_a = 1 + i * n_seg
        r_b = 1 + (i + 1) * n_seg
        for j in range(n_seg):
            jn = (j + 1) % n_seg
            faces.append([r_a + j, r_b + j, r_b + jn])
            faces.append([r_a + j, r_b + jn, r_a + jn])
    last = 1 + (n_rings - 1) * n_seg
    for j in range(n_seg):
        jn = (j + 1) % n_seg
        faces.append([last + j, last + jn, end_idx])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int32)


def create_sample_realistic_usd(usd_path: str) -> str:
    """Crea un archivo OpenUSD (.usda) anatómico multi-malla con materiales UsdPreviewSurface."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    usd_path = os.path.abspath(os.path.expanduser(usd_path))
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    if os.path.exists(usd_path):
        os.remove(usd_path)

    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, '/RealisticPerson')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, '/RealisticPerson/Looks')

    def make_mat(name: str, rgb: Tuple[float, float, float], rough: float = 0.6):
        mat = UsdShade.Material.Define(stage, f'/RealisticPerson/Looks/{name}')
        sh = UsdShade.Shader.Define(stage, f'/RealisticPerson/Looks/{name}/PreviewSurface')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(rough))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return mat

    mat_skin = make_mat('SkinMat', (0.89, 0.66, 0.52), 0.55)
    mat_hair = make_mat('HairMat', (0.16, 0.11, 0.08), 0.75)
    mat_jacket = make_mat('JacketMat', (0.10, 0.34, 0.80), 0.60)
    mat_shirt = make_mat('ShirtMat', (0.90, 0.92, 0.95), 0.70)
    mat_sleeve = make_mat('SleeveMat', (0.12, 0.24, 0.52), 0.65)
    mat_pants = make_mat('PantsMat', (0.18, 0.21, 0.26), 0.75)
    mat_shoes = make_mat('ShoeMat', (0.12, 0.12, 0.14), 0.50)

    parts = [
        ('TorsoJacket', *_tapered_limb_mesh((0.0, 0.0, 0.01), (0.0, 0.0, 0.33), (0.09, 0.155), (0.105, 0.195)), mat_jacket),
        ('ShirtCollar', *_tapered_limb_mesh((-0.01, 0.0, 0.31), (-0.01, 0.0, 0.355), (0.065, 0.085), (0.050, 0.060)), mat_shirt),
        ('Neck', *_tapered_limb_mesh((0.0, 0.0, 0.34), (0.0, 0.0, 0.40), (0.044, 0.046), (0.041, 0.043)), mat_skin),
        ('Head', *_ellipsoid_mesh((-0.005, 0.0, 0.475), (0.102, 0.088, 0.112)), mat_skin),
        ('HairCap', *_ellipsoid_mesh((0.012, 0.0, 0.505), (0.104, 0.092, 0.095)), mat_hair),
        ('PelvisHips', *_tapered_limb_mesh((0.0, 0.0, -0.07), (0.0, 0.0, 0.015), (0.088, 0.150), (0.090, 0.155)), mat_pants),
        ('LeftArmSleeve', *_tapered_limb_mesh((0.0, 0.195, 0.15), (0.0, 0.195, 0.30), (0.042, 0.044), (0.048, 0.050)), mat_sleeve),
        ('LeftForearm', *_tapered_limb_mesh((0.0, 0.195, 0.01), (0.0, 0.195, 0.15), (0.036, 0.038), (0.042, 0.044)), mat_skin),
        ('LeftHand', *_ellipsoid_mesh((0.0, 0.195, -0.03), (0.036, 0.028, 0.048)), mat_skin),
        ('RightArmSleeve', *_tapered_limb_mesh((0.0, -0.195, 0.15), (0.0, -0.195, 0.30), (0.042, 0.044), (0.048, 0.050)), mat_sleeve),
        ('RightForearm', *_tapered_limb_mesh((0.0, -0.195, 0.01), (0.0, -0.195, 0.15), (0.036, 0.038), (0.042, 0.044)), mat_skin),
        ('RightHand', *_ellipsoid_mesh((0.0, -0.195, -0.03), (0.036, 0.028, 0.048)), mat_skin),
        ('LeftLegThigh', *_tapered_limb_mesh((0.0, 0.095, -0.38), (0.0, 0.095, -0.04), (0.056, 0.058), (0.068, 0.072)), mat_pants),
        ('LeftCalf', *_tapered_limb_mesh((0.0, 0.095, -0.70), (0.0, 0.095, -0.38), (0.046, 0.048), (0.056, 0.058)), mat_pants),
        ('LeftShoe', *_ellipsoid_mesh((-0.035, 0.095, -0.725), (0.115, 0.052, 0.038)), mat_shoes),
        ('RightLegThigh', *_tapered_limb_mesh((0.0, -0.095, -0.38), (0.0, -0.095, -0.04), (0.056, 0.058), (0.068, 0.072)), mat_pants),
        ('RightCalf', *_tapered_limb_mesh((0.0, -0.095, -0.70), (0.0, -0.095, -0.38), (0.046, 0.048), (0.056, 0.058)), mat_pants),
        ('RightShoe', *_ellipsoid_mesh((-0.035, -0.095, -0.725), (0.115, 0.052, 0.038)), mat_shoes),
    ]

    for name, verts, faces, mat in parts:
        m_prim = UsdGeom.Mesh.Define(stage, f'/RealisticPerson/{name}')
        m_prim.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in verts])
        m_prim.CreateFaceVertexCountsAttr([3] * len(faces))
        m_prim.CreateFaceVertexIndicesAttr([int(idx) for tri in faces for idx in tri])
        m_prim.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(m_prim.GetPrim()).Bind(mat)

    stage.GetRootLayer().Save()
    return usd_path


def main():
    parser = argparse.ArgumentParser(
        description='Importa un archivo USD (.usd/.usda/.usdc/.usdz) al modelo Gazebo person_target.')
    parser.add_argument('usd_path', nargs='?', default='',
                        help='Ruta al archivo .usd/.usda/.usdc/.usdz (o --create-sample)')
    parser.add_argument('--model-dir', default='',
                        help='Directorio de person_target (por defecto el del paquete lite6_handeye)')
    parser.add_argument('--height', type=float, default=1.58,
                        help='Estatura objetivo de la persona en metros (por defecto 1.58)')
    parser.add_argument('--ground-z', type=float, default=-0.95,
                        help='Cota Z del suelo en el marco de person_target (por defecto -0.95)')
    parser.add_argument('--create-sample', action='store_true',
                        help='Genera primero el archivo USD realista de ejemplo y lo importa')
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir:
        try:
            from ament_index_python import get_package_share_directory
            model_dir = os.path.join(get_package_share_directory('lite6_handeye'), 'sim', 'models', 'person_target')
        except Exception:
            model_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', 'sim', 'models', 'person_target'))

    usd_path = args.usd_path
    if args.create_sample or not usd_path:
        usd_path = usd_path or os.path.join(model_dir, 'usd', 'realistic_person.usda')
        if not os.path.exists(usd_path) or args.create_sample:
            create_sample_realistic_usd(usd_path)

    info = import_usd_to_person_target(
        usd_path=usd_path,
        model_dir=model_dir,
        target_height=args.height,
        ground_z=args.ground_z,
    )
    print(f'[usd_person] Importado {info["usd_path"]} -> {info["sdf_path"]} '
          f'({info["submeshes"]} sub-mallas, {info["vertices"]} vértices, {info["triangles"]} triángulos)')


if __name__ == '__main__':
    main()
