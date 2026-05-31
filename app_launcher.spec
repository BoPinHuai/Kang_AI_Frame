# -*- mode: python ; coding: utf-8 -*-
#
# 打包命令（在项目根目录执行）：
#   pyinstaller app_launcher.spec --clean
#
# 输出目录：dist/KangAI/   压缩后分发即可。

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# ── 只精确收集必要包的数据文件 ────────────────────────────────────────────────
chroma_datas,  chroma_bins,  chroma_hidden  = collect_all('chromadb')
st_datas,      st_bins,      st_hidden      = collect_all('sentence_transformers')

all_datas = (
    chroma_datas + st_datas
    + collect_data_files('tokenizers')
    + collect_data_files('huggingface_hub')
    + [
        ('static',    'static'),
        ('providers', 'providers'),
    ]
)

all_binaries = chroma_bins + st_bins

all_hidden = (
    chroma_hidden + st_hidden
    + collect_submodules('uvicorn')
    + collect_submodules('fastapi')
    + collect_submodules('starlette')
    + collect_submodules('pydantic')
    + collect_submodules('anyio')
    + collect_submodules('multipart')
    + [
        # pywebview
        'webview', 'clr', 'pythonnet', 'bottle', 'proxy_tools',
        # 文档解析
        'pypdf', 'pypdf._reader', 'pypdf._writer', 'pypdf.filters',
        'docx', 'openpyxl', 'openpyxl.styles', 'openpyxl.utils',
        # LLM
        'ollama', 'openai', 'httpx', 'httpcore',
        # Python 3.13 built-ins 有时被 PyInstaller 遗漏
        'unicodedata', 'encodings', 'encodings.utf_8', 'encodings.ascii',
        'encodings.latin_1', 'encodings.idna', '_codecs',
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
    excludes=[
        # 不需要的大包，减小体积
        'matplotlib', 'notebook', 'IPython', 'scipy',
        'sklearn', 'pandas', 'PIL',
        'torch.distributed', 'tensorboard',
        'pytest', 'unittest',
    ],
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
    strip=False,
    upx=True,
    console=False,
    icon='static/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KangAI',
)
