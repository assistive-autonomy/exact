## ExAct Grammar

Programs follow a time-indexed format with sensor predicates:

```
[start,end]body.axis(value)*body.axis(value),[start,end]...
```

Example:
```
[0,500]head.z(1.4)*lhand.x(0.5),[500,1000]pelvis.y(0.8)
```

### Supported Body Parts (23 total)
- Core: `pelvis`, `torso`, `spine`, `chest`, `neck`, `head`
- Left leg: `lhip`, `lknee`, `lankle`, `ltoe`
- Right leg: `rhip`, `rknee`, `rankle`, `rtoe`
- Left arm: `lthorax`, `lshoulder`, `lelbow`, `lwrist`, `lhand`
- Right arm: `rthorax`, `rshoulder`, `relbow`, `rwrist`, `rhand`

### Axes
- `x`, `y`, `z` (69 total predicates = 23 body parts × 3 axes)