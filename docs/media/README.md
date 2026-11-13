# Media and demo clips

This folder is for short demo clips and animated previews. Video files are
**not** committed by default because they inflate the repository and GitHub
does not stream large files reliably from normal Git history.

## Recommended workflow

1. Record a 20-40 second clip at 1080p or 720p.
2. Trim dead time and keep the interesting segment (takeoff, search, detection).
3. Upload the clip to a release asset, YouTube, or another stable host.
4. Link it from the root README and from `docs/RESULTS.md`.

## If a clip must live in the repository

Use Git LFS:

```bash
git lfs install
git lfs track "docs/media/*.mp4"
git add .gitattributes docs/media/your-clip.mp4
git commit -m "docs(media): add demo clip"
```

Keep clips under 20 MB where possible. Do not commit raw simulator captures,
multi-gigabyte recordings, or copyrighted music.

## Suggested clips

| Clip | Content |
| --- | --- |
| `airsim-search-demo.mp4` | Manual or autonomous flight, victim detection, and HUD. |
| `pybullet-2c-search.mp4` | Full-occlusion search with fog-map coverage. |
| `training-timelapse.mp4` | Short TensorBoard or GUI timelapse (optional). |
