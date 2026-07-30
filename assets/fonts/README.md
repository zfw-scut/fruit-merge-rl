# Android / desktop shared UI font

`ui-nunito.fnt` and `ui-nunito.png` are generated at 64 px from Nunito weight
750. Android and the Windows preview both load this same atlas so typography
metrics remain identical.

The upstream variable font is stored at `source/Nunito[wght].ttf` and is
licensed under the SIL Open Font License 1.1 in `source/OFL.txt`.

- Upstream: `https://github.com/googlefonts/nunito`
- Source size: `276932` bytes
- Source SHA-256:
  `BB55A5CA5C2042335B3991AF27C4D0705D0EF41CAC6164AC737FD8F2A1E85207`

Regenerate the atlas with the local `python-torch` environment:

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run `
  -n python-torch python tools\generate_mobile_ui_font.py
```
