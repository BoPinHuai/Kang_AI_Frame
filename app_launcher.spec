# -*- mode: python ; coding: utf-8 -*-
#
# 打包命令（在项目根目录执行）：
#   pip install pyinstaller
#   pyinstaller app_launcher.spec
#
# 输出目录：dist/KangAI/
# 将整个 dist/KangAI/ 文件夹压缩分发即可。
#
# 注意：首次运行会自动下载 ML 模型（~200MB），需联网。
#       之后模型缓存在 hf-cache/ 里，离线可用。

from PyInstaller.utils.hooks import collect_all, collect_submodules
import sys

block_cipher = None

# ── 收集各包的动态数据 ────────────────────────────────────────────────────────
def collect(pkg):
    d, b, h = collect_all(pkg)
    return d, b, h

chroma_datas,   chroma_bins,   chroma_hidden   = collect('chromadb')
st_datas,       st_bins,       st_hidden       = collect('sentence_transformers')
torch_datas,    torch_bins,    torch_hidden     = collect('torch')
trf_datas,      trf_bins,      trf_hidden      = collect('transformers')

all_datas = (
    chroma_datas + st_datas + torch_datas + trf_datas
    + [
        ('static',    'static'),     # 前端 HTML/CSS/JS
        ('providers', 'providers'),  # LLM provider 模块
    ]
)
all_binaries = chroma_bins + st_bins + torch_bins + trf_bins
all_hidden   = (
    chroma_hidden + st_hidden + torch_hidden + trf_hidden
    + collect_submodules('uvicorn')
    + collect_submodules('fastapi')
    + collect_submodules('pydantic')
    + collect_submodules('anyio')
    + collect_submodules('starlette')
    + [
        'pypdf', 'docx', 'openpyxl', 'ollama', 'openai',
        'webview', 'clr', 'pythonnet',
    ]
)

a = Analysis(
    ['app_launcher.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=['matplotlib', 'notebook', 'IPython', 'scipy', 'sklearn'],
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
    name='KangAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # 不显示黑色终端窗口
    icon='static/icon.ico', # 任务栏图标
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KangAI',          # 输出到 dist/KangAI/
)
