# Evaluation Objects

This directory contains the 22 object instances used by the
`shadow_grasp_0725` dataset.

Each object uses the following layout:

```text
<object_id>/
  visual/simplified.obj
  collision/convex_piece_*.obj
```

- `visual/simplified.obj` is the render mesh.
- `collision/convex_piece_*.obj` contains the convex collision decomposition.
- Object scale, pose and placement are episode-level properties and are not
  baked into these meshes.
- `objects.csv` records dataset coverage and available collision pieces.

The meshes were copied from the locally processed DGN object assets. Confirm
the upstream dataset license before redistributing them publicly.
