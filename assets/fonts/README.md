# Android / desktop shared UI font

`ui-cute.fnt` and `ui-cute.png` are generated at 64 px from ZCOOL KuaiLe
(站酷快乐体). Android and the Windows preview both load this same atlas so
typography metrics remain identical. The atlas contains printable ASCII plus
the explicit Chinese mobile-UI character set; it does not pack the full CJK
font.

The upstream font is stored at `source/ZCOOLKuaiLe-Regular.ttf` and is
licensed under the SIL Open Font License 1.1 in
`source/ZCOOLKuaiLe-OFL.txt`. The Android build also places that license in
the APK's `assets/licenses/` directory.

- Official source:
  `https://github.com/google/fonts/tree/main/ofl/zcoolkuaile`
- Source size: `1514968` bytes
- Source SHA-256:
  `812A6FC1FE54B6D73A419245C32DFEBA8AA33104D5BE90D1CF6AF082007CB71D`

Regenerate the atlas with the local `python-torch` environment:

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run `
  -n python-torch python tools\generate_mobile_ui_font.py
```

The generator validates every requested code point against the upstream
font's Unicode cmap before writing the atlas.
