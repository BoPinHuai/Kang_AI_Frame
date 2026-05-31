"""
桌面客户端启动器
使用 pywebview 将 Web 应用包装为原生桌面窗口。

直接运行：
    python app_launcher.py

打包为 exe（需先 pip install pyinstaller）：
    pyinstaller app_launcher.spec
"""

import time
import threading
import urllib.request
from pathlib import Path

import webview
import uvicorn


# ── 配置 ──────────────────────────────────────────────────────────────────────
HOST          = "127.0.0.1"
PORT          = 8000
WINDOW_TITLE  = "知识库"
WINDOW_WIDTH  = 1360
WINDOW_HEIGHT = 860
WINDOW_MIN_W  = 900
WINDOW_MIN_H  = 600
BASE_DIR      = Path(__file__).parent
ICON_PATH     = BASE_DIR / "static" / "icon.ico"

_window: webview.Window | None = None


# ── 暴露给前端 JS 的窗口控制 API ──────────────────────────────────────────────
class WindowApi:
    def minimize(self):
        if _window: _window.minimize()

    def toggle_maximize(self):
        if _window: _window.toggle_fullscreen()

    def close(self):
        if _window: _window.destroy()


# ── FastAPI 服务器 ─────────────────────────────────────────────────────────────
def _run_server():
    from api import app
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def _wait_for_server(timeout: int = 30) -> bool:
    url = f"http://{HOST}:{PORT}/api/settings"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


# ── Windows 原生调整（图标 + 恢复调整大小边框）────────────────────────────────
def _apply_win32_tweaks():
    """在后台线程里等窗口出现，然后：
    1. 把 WS_THICKFRAME 加回 WS_POPUP 窗口 → 恢复拖拽调整大小
    2. 替换任务栏 / 窗口图标为自定义 .ico
    """
    time.sleep(1.2)   # 等窗口初始化完成
    try:
        import ctypes
        import ctypes.wintypes

        # ── 找到窗口句柄 ──
        hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)
        if not hwnd:
            return

        # ── 1. 恢复调整大小边框 ──
        GWL_STYLE    = -16
        WS_THICKFRAME = 0x00040000
        SWP_FLAGS     = 0x0027   # NOMOVE | NOSIZE | NOZORDER | FRAMECHANGED

        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style | WS_THICKFRAME)
        ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_FLAGS)

        # ── 2. 替换图标 ──
        if ICON_PATH.exists():
            LR_LOADFROMFILE = 0x10
            hicon = ctypes.windll.user32.LoadImageW(
                None, str(ICON_PATH), 1, 0, 0, LR_LOADFROMFILE)
            if hicon:
                WM_SETICON = 0x0080
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 1, hicon)  # 大图标
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 0, hicon)  # 小图标

    except Exception:
        pass  # 非 Windows 环境静默跳过


# ── 加载中页面 ────────────────────────────────────────────────────────────────
_LOADING_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box;}
body{display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:100vh;background:#faf9f5;
  font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;}
.spinner{width:32px;height:32px;border:3px solid #e7e3d8;border-top-color:#d97757;
  border-radius:50%;animation:spin .7s linear infinite;margin-bottom:14px;}
@keyframes spin{to{transform:rotate(360deg);}}
p{font-size:13px;color:#83807a;}
</style></head>
<body><div class="spinner"></div><p>正在启动知识库…</p></body></html>"""

_TIMEOUT_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body{display:flex;align-items:center;justify-content:center;height:100vh;
  background:#faf9f5;font-family:sans-serif;color:#c0392b;font-size:14px;}
</style></head><body><p>⚠️ 服务器启动超时，请关闭后重试。</p></body></html>"""


# ── 主入口 ────────────────────────────────────────────────────────────────────
def main():
    global _window

    # 1. 后台启动 FastAPI
    threading.Thread(target=_run_server, daemon=True).start()

    # 2. 创建无边框窗口
    _window = webview.create_window(
        WINDOW_TITLE,
        html=_LOADING_HTML,
        js_api=WindowApi(),
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(WINDOW_MIN_W, WINDOW_MIN_H),
        frameless=True,
        text_select=True,
    )

    # 3. 等服务器就绪后跳转；同时后台修复 Win32 样式
    def _on_shown():
        threading.Thread(target=_apply_win32_tweaks, daemon=True).start()
        if _wait_for_server(timeout=30):
            # ?desktop=1 让前端知道自己在客户端里，显示自定义标题栏
            _window.load_url(f"http://{HOST}:{PORT}?desktop=1")
        else:
            _window.load_html(_TIMEOUT_HTML)

    webview.start(_on_shown, debug=False)


if __name__ == "__main__":
    main()
