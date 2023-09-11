# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['facetracker.py'],
    pathex=[],
    binaries=[('dshowcapture/*.dll', '.'), ('escapi/*.dll', '.'), ('venv/lib/site-packages/onnxruntime/capi/*.dll', 'onnxruntime\\capi'), ('venv/lib/site-packages/mediapipe/python/*.dll', 'mediapipe\\python'), ('msvcp140.dll', '.'), ('vcomp140.dll', '.'), ('concrt140.dll', '.'), ('vccorlib140.dll', '.'), ('run.bat', '.')],
    datas=[('mediapipe/modules/hand_landmark', '.'), ('mediapipe/modules/hand_landmark', '.'), ('mediapipe/modules/palm_detection', '.'), ('mediapipe/modules/palm_detection', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='facetracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='facetracker',
)
