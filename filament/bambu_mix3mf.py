#!/usr/bin/env python3
"""Build a Bambu-Studio *Color Mixing* 3MF from parts with per-part filament or
mix-ratio assignments.

Reusable across the toolchain: the mixture calibration pad tags 11 pads with
ratios, and the stained-glass model generator will tag each pane with the recipe
the solver picked -- both just hand this module a list of parts.

A Bambu color-mix is stored as extra "virtual" filament slots in
``project_settings.config`` (``filament_is_mixed`` / ``filament_mixed_components``
/ ``filament_mixed_sublayer_ratios`` + ``enable_mixed_color_sublayer``), with each
object/part pointing at a slot via ``extruder``.  We template a REAL Bambu export
so printer + filament settings match the user's machine, then:
  * keep the physical BASE filaments as slots 1..B (e.g. A, B, black),
  * de-duplicate the distinct mixes into virtual slots B+1.., and
  * write the geometry as one object whose parts each carry their slot.

API:
    write_bambu_color_mix_3mf(out_path, template_3mf, bases, parts)
      bases: [{"colour": "#RRGGBB", "colour_type"?: "2"}, ...]   # slots 1..B
      parts: [{"name", "boxes"|"mesh", "slot": int   # a base slot (1-based)
                                       | "mix": {"components":[slot,...],
                                                 "ratios":[float,...],
                                                 "colour"?: "#RRGGBB"}}, ...]
    -> {"n_slots", "mixes", "extruder_of_part"}
"""
import copy
import json
import os
import re
import sys
import uuid
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_calibration_pad import _boxes_to_mesh  # noqa: E402


def _uuid(seed):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "sgg-mix3mf-" + str(seed)))


def _mesh_of(part):
    if "mesh" in part and part["mesh"] is not None:
        return part["mesh"]
    return _boxes_to_mesh(part["boxes"])


def _mix_key(mix):
    return (tuple(mix["components"]),
            tuple(round(float(r), 4) for r in mix["ratios"]))


def _object_xml(oid, verts, tris):
    vs = "".join('<vertex x="%.6f" y="%.6f" z="%.6f"/>' % tuple(v) for v in verts)
    ts = "".join('<triangle v1="%d" v2="%d" v3="%d"/>' % tuple(t) for t in tris)
    return ('<object id="%d" p:UUID="%s" type="model"><mesh><vertices>%s'
            '</vertices><triangles>%s</triangles></mesh></object>'
            % (oid, _uuid("obj%d" % oid), vs, ts))


def _extend_config(template_cfg, bases, mixes):
    """Template project_settings.config to len(bases)+len(mixes) filament slots.

    Per-filament arrays come in two shapes: length n (one value per filament) and
    length m*n (m values per filament, interleaved -- e.g. m=2 for the two
    extruder variants on a dual-nozzle machine).  Both are extended per filament,
    copying filament 0's block for the new slots.  flush_volumes_matrix (n*n) is
    rebuilt, and filament_self_index is renumbered."""
    cfg = copy.deepcopy(template_cfg)
    n_old = len(cfg["filament_type"])
    n = len(bases) + len(mixes)

    def blocks(v, m):                                    # per-filament m-blocks
        out = []
        for i in range(n):
            s = i if i < n_old else 0
            out += list(v[s * m:s * m + m])
        return out

    for k, v in list(cfg.items()):
        if not isinstance(v, list):
            continue
        if len(v) == n_old:                              # one value per filament
            cfg[k] = [v[i] if i < n_old else v[0] for i in range(n)]
        elif k.startswith("filament_") and len(v) > n_old and len(v) % n_old == 0:
            cfg[k] = blocks(v, len(v) // n_old)          # m values per filament
    if "filament_self_index" in template_cfg:            # self index = slot number
        m = len(template_cfg["filament_self_index"]) // n_old
        cfg["filament_self_index"] = [str(i + 1) for i in range(n) for _ in range(m)]
    if len(cfg.get("flush_volumes_matrix", [])) == n_old * n_old:
        cfg["flush_volumes_matrix"] = ["0" if i == j else "280"
                                       for i in range(n) for j in range(n)]
    fv = cfg.get("flush_volumes_vector")
    if isinstance(fv, list) and len(fv) % n_old == 0 and fv:
        cfg["flush_volumes_vector"] = blocks(fv, len(fv) // n_old)
    colours = [b["colour"] for b in bases] + [m.get("colour", "#888888")
                                              for m in mixes]
    cfg["enable_mixed_color_sublayer"] = "1"
    cfg["filament_colour"] = colours
    cfg["filament_multi_colour"] = colours
    cfg["default_filament_colour"] = [""] * n
    cfg["filament_colour_type"] = ([b.get("colour_type", "2") for b in bases]
                                   + ["1"] * len(mixes))
    cfg["filament_is_mixed"] = ["0"] * len(bases) + ["1"] * len(mixes)
    cfg["filament_mixed_components"] = ([""] * len(bases)
        + [",".join(str(c) for c in m["components"]) for m in mixes])
    cfg["filament_mixed_sublayer_ratios"] = ([""] * len(bases)
        + [",".join("%.4f" % r for r in m["ratios"]) for m in mixes])
    cfg["filament_mixed_gradient"] = ["0"] * n
    cfg["filament_mixed_gradient_curve"] = [""] * n
    cfg["filament_mixed_gradient_per_part"] = ["0"] * n
    cfg["filament_mixed_gradient_range"] = [""] * n
    return cfg


def write_bambu_color_mix_3mf(out_path, template_3mf, bases, parts, bed_mm=256.0):
    """See module docstring.  Returns a summary dict."""
    with zipfile.ZipFile(template_3mf) as z:
        tpl = {n: z.read(n) for n in z.namelist()}
    template_cfg = json.loads(tpl["Metadata/project_settings.config"])

    # de-duplicate mixes -> virtual slots after the bases
    mixes, mix_slot = [], {}
    for part in parts:
        if "mix" in part and part["mix"] is not None:
            key = _mix_key(part["mix"])
            if key not in mix_slot:
                mix_slot[key] = len(bases) + 1 + len(mixes)
                mixes.append(part["mix"])
    cfg = _extend_config(template_cfg, bases, mixes)
    n_slots = len(bases) + len(mixes)

    # meshes + per-part extruder + bounding box (for centring on the bed)
    meshes, extruder_of = [], []
    lo = [1e18, 1e18]
    hi = [-1e18, -1e18]
    for part in parts:
        verts, tris = _mesh_of(part)
        meshes.append((verts, tris))
        for x, y, _z in verts:
            lo[0], lo[1] = min(lo[0], x), min(lo[1], y)
            hi[0], hi[1] = max(hi[0], x), max(hi[1], y)
        if "mix" in part and part["mix"] is not None:
            extruder_of.append(mix_slot[_mix_key(part["mix"])])
        else:
            extruder_of.append(int(part["slot"]))
    tx = bed_mm / 2.0 - (lo[0] + hi[0]) / 2.0
    ty = bed_mm / 2.0 - (lo[1] + hi[1]) / 2.0

    part_xml, comps, ms_parts = [], [], []
    for k, (part, (verts, tris)) in enumerate(zip(parts, meshes), start=1):
        part_xml.append(_object_xml(k, verts, tris))
        comps.append('<component p:path="/3D/Objects/object_1.model" objectid="%d" '
                     'p:UUID="%s" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>'
                     % (k, _uuid("comp%d" % k)))
        ms_parts.append(
            '  <part id="%d" subtype="normal_part">\n'
            '   <metadata key="name" value="%s"/>\n'
            '   <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
            '   <metadata key="extruder" value="%d"/>\n'
            '   <mesh_stat face_count="%d" edges_fixed="0" degenerate_facets="0" '
            'facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
            '  </part>' % (k, part["name"], extruder_of[k - 1], len(tris)))
    asm = len(parts) + 1

    core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    bbs = "http://schemas.bambulab.com/package/2021"
    prod = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
    obj_model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" xmlns="%s" xmlns:BambuStudio='
        '"%s" xmlns:p="%s" requiredextensions="p">\n'
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n'
        '  %s\n </resources>\n <build/>\n</model>\n'
        % (core, bbs, prod, "\n  ".join(part_xml)))
    # preserve the template's Bambu metadata block (Application=BambuStudio-...,
    # 3mfVersion, etc.) verbatim -- Bambu checks it to accept the file as its own.
    tmeta = re.findall(r'<metadata name=.*?</metadata>',
                       tpl["3D/3dmodel.model"].decode())
    meta_block = "\n ".join(tmeta) or '<metadata name="Application">BambuStudio</metadata>'
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" xmlns="%s" xmlns:BambuStudio='
        '"%s" xmlns:p="%s" requiredextensions="p">\n %s\n <resources>\n'
        '  <object id="%d" p:UUID="%s" type="model">\n   <components>\n   %s\n'
        '   </components>\n  </object>\n </resources>\n <build p:UUID="%s">\n'
        '  <item objectid="%d" p:UUID="%s" transform="1 0 0 0 1 0 0 0 1 %.6f %.6f '
        '0" printable="1"/>\n </build>\n</model>\n'
        % (core, bbs, prod, meta_block, asm, _uuid("asm"), "\n   ".join(comps),
           _uuid("build"), asm, _uuid("item"), tx, ty))
    model_settings = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n <object id="%d">\n'
        '  <metadata key="name" value="model"/>\n'
        '  <metadata key="extruder" value="1"/>\n%s\n </object>\n <plate>\n'
        '  <metadata key="plater_id" value="1"/>\n'
        '  <metadata key="plater_name" value=""/>\n'
        '  <metadata key="locked" value="false"/>\n'
        '  <metadata key="filament_map_mode" value="Auto For Flush"/>\n'
        '  <model_instance>\n   <metadata key="object_id" value="%d"/>\n'
        '   <metadata key="instance_id" value="0"/>\n  </model_instance>\n'
        ' </plate>\n <assemble>\n  <assemble_item object_id="%d" instance_id="0" '
        'transform="1 0 0 0 1 0 0 0 1 %.6f %.6f 0" offset="0 0 0"/>\n </assemble>\n'
        '</config>\n' % (asm, "\n".join(ms_parts), asm, asm, tx, ty))
    obj_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://'
        'schemas.openxmlformats.org/package/2006/relationships">\n <Relationship '
        'Target="/3D/Objects/object_1.model" Id="rel-1" Type="http://schemas.'
        'microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>\n')

    # start from ALL of the template's members (thumbnails, cut_information,
    # [Content_Types], _rels/.rels with its Bambu thumbnail relationships, ...)
    # and override only what we regenerate -- dropping any of these makes Bambu
    # treat the file as a foreign 3MF and ignore project_settings.config.
    out = dict(tpl)
    out["3D/3dmodel.model"] = model.encode()
    out["3D/_rels/3dmodel.model.rels"] = obj_rels.encode()
    out["3D/Objects/object_1.model"] = obj_model.encode()
    out["Metadata/project_settings.config"] = json.dumps(cfg, indent=4).encode()
    out["Metadata/model_settings.config"] = model_settings.encode()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in out.items():
            z.writestr(name, data)
    return {"n_slots": n_slots, "mixes": mixes, "extruder_of_part": extruder_of}
