from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser


def enable_dpi_awareness() -> None:
    """Enable per-monitor DPI awareness before Tk creates its root window."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


enable_dpi_awareness()

import tkinter as tk
from tkinter import messagebox, ttk


CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
ERROR_ALREADY_EXISTS = 183
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent.parent
    return Path(__file__).resolve().parent


ROOT = project_root()
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "launcher.log",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("literature-rag-launcher")


def public_env() -> dict[str, str]:
    """Read only non-secret launcher settings; never log or return credentials."""
    allowed = {"RAG_API_HOST", "RAG_API_PORT", "RAG_UI_PORT", "RAG_EVAL_PORT"}
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        return values
    try:
        for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in allowed:
                values[key] = value.strip().strip('"').strip("'")
    except OSError:
        logger.warning("Unable to read public launcher settings")
    return values


PUBLIC_ENV = public_env()
API_HOST = PUBLIC_ENV.get("RAG_API_HOST", "127.0.0.1")


def build_manifest() -> dict[str, str]:
    defaults = {
        "project_id": "literature-rag-eval-code",
        "application_version": "2.0.0",
        "build_id": "20260807-rag-accuracy-v2",
        "prompt_version": "rag_answer_prompt_v2",
    }
    try:
        raw = json.loads((ROOT / "build_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return defaults
    return {key: str(raw.get(key) or value) for key, value in defaults.items()}


EXPECTED_BUILD = build_manifest()


def configured_port(key: str, default: int) -> int:
    try:
        value = int(PUBLIC_ENV.get(key, str(default)))
        return value if 1 <= value <= 65535 else default
    except ValueError:
        return default


API_PORT = configured_port("RAG_API_PORT", 8010)
UI_PORT = configured_port("RAG_UI_PORT", 8501)
EVAL_PORT = configured_port("RAG_EVAL_PORT", 8502)
API_BASE = f"http://{API_HOST}:{API_PORT}"

PID_FILES = {
    "api": ROOT / ".rag_api.pid",
    "ui": ROOT / ".rag_ui.pid",
    "eval": ROOT / ".rag_eval.pid",
}


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    def as_int(self) -> int:
        return (self.dwHighDateTime << 32) | self.dwLowDateTime


def process_identity(pid: int) -> tuple[str, int] | None:
    if sys.platform != "win32" or pid <= 0:
        return None
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        creation = FILETIME()
        exit_time = FILETIME()
        kernel_time = FILETIME()
        user_time = FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
        return str(Path(buffer.value).resolve()).casefold(), creation.as_int()
    finally:
        kernel32.CloseHandle(handle)


def read_pid_record(kind: str) -> dict | None:
    try:
        record = json.loads(PID_FILES[kind].read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def pid_record_current(kind: str) -> bool:
    record = read_pid_record(kind) or {}
    return (
        record.get("build_id") == EXPECTED_BUILD["build_id"]
        and record.get("prompt_version") == EXPECTED_BUILD["prompt_version"]
        and str(record.get("working_directory", "")).casefold() == str(ROOT).casefold()
    )


def owned_process(kind: str) -> tuple[bool, int | None]:
    record = read_pid_record(kind)
    if not record:
        return False, None
    try:
        if str(record.get("project_root", "")).casefold() != str(ROOT).casefold():
            return False, None
        if record.get("service") != kind:
            return False, None
        pid = int(record["pid"])
        expected_identity = (str(record["executable"]).casefold(), int(record["created_at"]))
    except (KeyError, TypeError, ValueError):
        return False, None
    current = process_identity(pid)
    return current == expected_identity, pid


def port_open(port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def request_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data if isinstance(data, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def api_is_current_project() -> bool:
    data = request_json(f"{API_BASE}/health")
    return health_identity(data) == "current"


def health_identity(data: dict | None) -> str:
    """Classify API identity without trusting port reachability alone."""
    if not data:
        return "unavailable"
    if data.get("project_id") != EXPECTED_BUILD["project_id"]:
        if data.get("service") == "literature-rag-api" and not data.get("project_id"):
            return "legacy_project"
        return "foreign"
    if (
        data.get("build_id") != EXPECTED_BUILD["build_id"]
        or data.get("prompt_version") != EXPECTED_BUILD["prompt_version"]
        or data.get("application_version") != EXPECTED_BUILD["application_version"]
    ):
        return "old_project"
    return "current"


def _powershell_json(command: str):
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        return json.loads(completed.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def listener_pid(port: int) -> int | None:
    raw = _powershell_json(
        f"Get-NetTCPConnection -LocalPort {int(port)} -State Listen "
        "-ErrorAction SilentlyContinue | Select-Object -First 1 "
        "-ExpandProperty OwningProcess | ConvertTo-Json"
    )
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def process_snapshot(pid: int) -> dict | None:
    raw = _powershell_json(
        f"Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}' | "
        "Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine,CreationDate | "
        "ConvertTo-Json -Compress"
    )
    return raw if isinstance(raw, dict) else None


def verified_project_tree_root(kind: str, port: int) -> tuple[int, dict] | None:
    """Resolve a listener to this project's venv parent without broad PID matching."""
    pid = listener_pid(port)
    if pid is None:
        return None
    expected_marker = (
        "-m uvicorn api_server:app" if kind == "api" else "-m streamlit run app.py"
    )
    listener = process_snapshot(pid)
    listener_command = str((listener or {}).get("CommandLine") or "").casefold()
    if expected_marker.casefold() not in listener_command:
        return None
    expected_python = str((ROOT / ".venv" / "Scripts" / "python.exe").resolve()).casefold()
    current = listener
    for _ in range(8):
        if not current:
            break
        executable = str(current.get("ExecutablePath") or "").casefold()
        command_line = str(current.get("CommandLine") or "").casefold()
        if executable == expected_python and expected_marker.casefold() in command_line:
            return int(current["ProcessId"]), current
        try:
            parent_pid = int(current.get("ParentProcessId") or 0)
        except (TypeError, ValueError):
            break
        if parent_pid <= 0:
            break
        current = process_snapshot(parent_pid)
    return None


def wait_port_released(port: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_open(port):
            return True
        time.sleep(0.25)
    return not port_open(port)


def stop_service(kind: str, *, quiet: bool = False) -> bool:
    """Stop only a PID-record-owned or independently verified project tree."""
    ports = {"api": API_PORT, "ui": UI_PORT, "eval": EVAL_PORT}
    port = ports[kind]
    owned, pid = owned_process(kind)
    root_pid = pid if owned else None
    verification = "pid_record"

    if root_pid is None and port_open(port):
        verified = verified_project_tree_root(kind, port)
        if verified:
            root_pid, snapshot = verified
            verification = "verified_command_tree"
            logger.info(
                "Recovered %s ownership without PID record: pid=%s command=%s",
                kind,
                root_pid,
                str(snapshot.get("CommandLine") or "")[:300],
            )
    if root_pid is None:
        record = read_pid_record(kind)
        if record:
            try:
                recorded_pid = int(record.get("pid", -1))
            except (TypeError, ValueError):
                recorded_pid = -1
            if process_identity(recorded_pid) is None:
                PID_FILES[kind].unlink(missing_ok=True)
        if port_open(port) and not quiet:
            raise RuntimeError(
                f"拒绝停止 {kind}：端口 {port} 的进程无法确认属于当前项目。"
            )
        return False

    result = subprocess.run(
        ["taskkill", "/PID", str(root_pid), "/T", "/F"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )
    if result.returncode not in (0, 128):
        raise RuntimeError(f"停止 {kind} 失败，请查看 launcher.log。")
    if not wait_port_released(port):
        raise RuntimeError(f"{kind} 进程已请求停止，但端口 {port} 未释放。")
    PID_FILES[kind].unlink(missing_ok=True)
    logger.info(
        "Stopped %s process tree pid=%s verification=%s build=%s",
        kind,
        root_pid,
        verification,
        EXPECTED_BUILD["build_id"],
    )
    return True


def offline_project_status() -> tuple[int, bool, bool, str]:
    pdf_dir = ROOT / "data" / "pdfs"
    pdf_count = len(list(pdf_dir.rglob("*.pdf"))) if pdf_dir.is_dir() else 0
    index_exists = (ROOT / "chroma_db" / "chroma.sqlite3").is_file()
    llm_configured = False
    llm_model = "unknown"
    env_file = ROOT / ".env"
    if env_file.is_file():
        try:
            for raw_line in env_file.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                if key.strip() == "OPENAI_API_KEY":
                    llm_configured = bool(value)
                elif key.strip() == "LLM_MODEL" and value:
                    llm_model = value
        except OSError:
            pass
    return pdf_count, index_exists, llm_configured, llm_model


def streamlit_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/_stcore/health", timeout=1.5
        ) as response:
            return response.status == 200 and b"ok" in response.read(32).lower()
    except (OSError, urllib.error.URLError):
        return False


class Launcher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.operation_lock = threading.Lock()
        self.refresh_running = False
        self.buttons: list[ttk.Button] = []

        root.title("Literature RAG")
        dpi = 96
        try:
            dpi = ctypes.windll.user32.GetDpiForWindow(root.winfo_id())
            if not dpi:
                dpi = ctypes.windll.user32.GetDpiForSystem()
        except Exception:
            pass
        self.dpi_scale = max(1.0, dpi / 96.0)
        root.tk.call("tk", "scaling", max(1.0, dpi / 72.0))
        root.geometry(f"{round(760 * self.dpi_scale)}x{round(620 * self.dpi_scale)}")
        root.minsize(round(700 * self.dpi_scale), round(570 * self.dpi_scale))
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        default_font = ("Microsoft YaHei UI", 10)
        title_font = ("Microsoft YaHei UI", 19, "bold")
        style = ttk.Style()
        style.configure("TLabel", font=default_font)
        style.configure("TButton", font=default_font, padding=(11, 8))
        style.configure("Title.TLabel", font=title_font)
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 10, "bold"))

        outer = ttk.Frame(root, padding=24)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Literature RAG", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="本地科学文献问答与评测 Production Demo",
            foreground="#555555",
        ).pack(anchor="w", pady=(3, 18))

        status_frame = ttk.LabelFrame(outer, text="服务状态", padding=14)
        status_frame.pack(fill="x")
        self.status_vars = {
            "api": tk.StringVar(value="未检查"),
            "ui": tk.StringVar(value="未检查"),
            "knowledge": tk.StringVar(value="未检查"),
            "index": tk.StringVar(value="未检查"),
            "embedding": tk.StringVar(value="未检查"),
            "version": tk.StringVar(value="未检查"),
            "llm": tk.StringVar(value="未检查"),
        }
        labels = [
            ("后端服务", "api"),
            ("前端服务", "ui"),
            ("知识库", "knowledge"),
            ("向量索引", "index"),
            ("Embedding", "embedding"),
            ("构建 / Prompt", "version"),
            ("LLM 配置", "llm"),
        ]
        for row, (label, key) in enumerate(labels):
            ttk.Label(status_frame, text=label, width=14).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Label(status_frame, textvariable=self.status_vars[key], style="Status.TLabel").grid(
                row=row, column=1, sticky="w", pady=4
            )
        status_frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(outer)
        button_frame.pack(fill="x", pady=(18, 12))
        actions = [
            ("启动项目", lambda: self.run_async(self.start_project)),
            ("停止项目", lambda: self.run_async(self.stop_project)),
            ("打开 RAG 页面", lambda: webbrowser.open(f"http://127.0.0.1:{UI_PORT}")),
            ("打开 API 文档", lambda: webbrowser.open(f"{API_BASE}/docs")),
            ("健康检查", lambda: self.refresh_async(show_message=True)),
            ("知识库状态", self.show_knowledge_status),
            ("打开评测看板", lambda: self.run_async(self.open_eval_dashboard)),
            ("打开日志目录", self.open_logs),
        ]
        for index, (label, command) in enumerate(actions):
            button = ttk.Button(button_frame, text=label, command=command)
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=5, pady=5)
            self.buttons.append(button)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.message_var = tk.StringVar(value="启动器已就绪。关闭窗口不会停止服务。")
        message_frame = ttk.LabelFrame(outer, text="最近状态", padding=12)
        message_frame.pack(fill="both", expand=True)
        ttk.Label(
            message_frame,
            textvariable=self.message_var,
            wraplength=round(670 * self.dpi_scale),
            justify="left",
        ).pack(anchor="w", fill="x")

        ttk.Label(
            outer,
            text=f"API {API_HOST}:{API_PORT}  ·  RAG UI 127.0.0.1:{UI_PORT}  ·  Eval 127.0.0.1:{EVAL_PORT}",
            foreground="#666666",
        ).pack(anchor="w", pady=(10, 0))

        self.refresh_async()
        self.root.after(15000, self.periodic_refresh)

    def set_message(self, message: str) -> None:
        self.root.after(0, self.message_var.set, message)

    def set_buttons(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.root.after(0, lambda: [button.configure(state=state) for button in self.buttons])

    def set_status(self, key: str, value: str) -> None:
        self.root.after(0, self.status_vars[key].set, value)

    def run_async(self, action) -> None:
        if not self.operation_lock.acquire(blocking=False):
            self.set_message("已有操作正在进行，请稍候。")
            return
        self.set_buttons(False)

        def worker() -> None:
            try:
                action()
            except Exception as exc:
                logger.error("Operation failed (%s): %s", type(exc).__name__, exc)
                self.set_message(str(exc))
                self.root.after(0, lambda: messagebox.showerror("Literature RAG", str(exc)))
            finally:
                self.set_buttons(True)
                self.operation_lock.release()
                self.refresh_async()

        threading.Thread(target=worker, daemon=True).start()

    def launch_process(self, kind: str, args: list[str], log_name: str) -> int:
        python = ROOT / ".venv" / "Scripts" / "python.exe"
        if not python.is_file():
            raise RuntimeError("项目虚拟环境不存在：请先按 README 安装依赖。")
        log_path = LOG_DIR / log_name
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        try:
            process = subprocess.Popen(
                [str(python), *args],
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
            )
        finally:
            log_handle.close()
        identity = None
        for _ in range(20):
            identity = process_identity(process.pid)
            if identity:
                break
            time.sleep(0.05)
        if not identity:
            raise RuntimeError(f"{kind} 进程启动后立即退出，请查看日志。")
        record = {
            "pid": process.pid,
            "executable": identity[0],
            "created_at": identity[1],
            "project_root": str(ROOT),
            "working_directory": str(ROOT),
            "service": kind,
            "command": args,
            "build_id": EXPECTED_BUILD["build_id"],
            "prompt_version": EXPECTED_BUILD["prompt_version"],
        }
        PID_FILES[kind].write_text(json.dumps(record), encoding="utf-8")
        logger.info("Started %s pid=%s", kind, process.pid)
        return process.pid

    def wait_for(self, check, timeout: float, label: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if check():
                return
            time.sleep(0.5)
        raise RuntimeError(f"{label}健康检查超时，请查看 logs 目录。")

    def wait_for_rag_warmup(self, timeout: float = 120.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            health = request_json(f"{API_BASE}/health", timeout=5)
            if health_identity(health) == "current":
                self.apply_health(health)
                embedding_status = (health.get("embedding") or {}).get("status")
                chroma_status = (health.get("vector_index") or {}).get("status")
                self.set_message(
                    f"正在预热：Embedding {embedding_status} · Chroma {chroma_status}"
                )
                if health.get("prewarmed") and health.get("rag_ready"):
                    return health
                if embedding_status == "failed" or chroma_status == "failed":
                    raise RuntimeError(
                        f"预热失败：Embedding {embedding_status} · Chroma {chroma_status}。请查看日志。"
                    )
            time.sleep(0.5)
        raise RuntimeError("Embedding / Chroma 预热超时，请查看 logs 目录。")

    def start_project(self) -> None:
        self.set_message("正在启动后端…")
        started: list[str] = []
        try:
            api_owned, _ = owned_process("api")
            api_port_used = port_open(API_PORT)
            if api_port_used:
                health = request_json(f"{API_BASE}/health")
                identity = health_identity(health)
                if identity == "current":
                    self.set_status("api", "已启动")
                elif identity in {"legacy_project", "old_project"}:
                    if not verified_project_tree_root("api", API_PORT):
                        raise RuntimeError(
                            "检测到旧版 Literature RAG，但无法验证进程树归属，未自动停止。"
                        )
                    self.set_status("api", "检测到旧版本服务")
                    self.set_message("检测到旧版本服务，正在安全停止 UI/API 完整进程树…")
                    stop_service("ui", quiet=True)
                    stop_service("api")
                    api_port_used = False
                    api_owned = False
                else:
                    raise RuntimeError(f"端口 {API_PORT} 被其他程序占用，未启动后端。")
            if not api_port_used and api_owned:
                raise RuntimeError("已有本项目后端进程，但健康检查失败；请先停止项目并查看日志。")
            if not api_port_used:
                if port_open(API_PORT):
                    raise RuntimeError(f"端口 {API_PORT} 尚未释放，拒绝启动新版本。")
                self.launch_process(
                    "api",
                    ["-m", "uvicorn", "api_server:app", "--host", API_HOST, "--port", str(API_PORT)],
                    "rag_api.log",
                )
                started.append("api")
                self.wait_for(api_is_current_project, 45, "后端")

            self.set_message("后端已就绪，正在启动前端…")
            ui_owned, _ = owned_process("ui")
            ui_port_used = port_open(UI_PORT)
            if ui_port_used:
                if streamlit_ready(UI_PORT) and ui_owned and pid_record_current("ui"):
                    self.set_status("ui", "已启动")
                elif verified_project_tree_root("ui", UI_PORT):
                    self.set_message("检测到旧版前端，正在安全重启…")
                    stop_service("ui")
                    ui_port_used = False
                    ui_owned = False
                else:
                    raise RuntimeError(f"端口 {UI_PORT} 被其他程序占用，未启动前端。")
            if not ui_port_used and ui_owned:
                raise RuntimeError("已有本项目前端进程，但健康检查失败；请先停止项目并查看日志。")
            if not ui_port_used:
                self.launch_process(
                    "ui",
                    [
                        "-m", "streamlit", "run", "app.py",
                        "--server.address", "127.0.0.1",
                        "--server.port", str(UI_PORT),
                        "--server.headless", "true",
                    ],
                    "rag_ui.log",
                )
                started.append("ui")
                self.wait_for(lambda: streamlit_ready(UI_PORT), 60, "前端")

            self.set_message("前端已就绪，正在预热 Embedding 与 Chroma…")
            health = self.wait_for_rag_warmup()
            self.apply_health(health)
            self.set_message("启动成功，正在打开 RAG 页面。")
            webbrowser.open(f"http://127.0.0.1:{UI_PORT}")
        except Exception:
            for kind in reversed(started):
                self.stop_kind(kind, quiet=True)
            raise

    def stop_kind(self, kind: str, *, quiet: bool = False) -> bool:
        return stop_service(kind, quiet=quiet)

    def stop_project(self) -> None:
        stopped: list[str] = []
        for kind in ("eval", "ui", "api"):
            if self.stop_kind(kind):
                stopped.append(kind)
        if stopped:
            self.set_message("已停止完整进程树：" + ", ".join(stopped))
        else:
            self.set_message("没有由本启动器管理的运行中服务；未终止其他程序。")

    def open_eval_dashboard(self) -> None:
        owned, _ = owned_process("eval")
        eval_port_used = port_open(EVAL_PORT)
        if eval_port_used:
            if not (owned and streamlit_ready(EVAL_PORT)):
                raise RuntimeError(f"端口 {EVAL_PORT} 被其他程序占用，未启动评测看板。")
        else:
            if owned:
                raise RuntimeError("已有本项目评测进程，但健康检查失败；请先停止项目并查看日志。")
            self.launch_process(
                "eval",
                [
                    "-m", "streamlit", "run", "app_eval.py",
                    "--server.address", "127.0.0.1",
                    "--server.port", str(EVAL_PORT),
                    "--server.headless", "true",
                ],
                "rag_eval.log",
            )
            self.wait_for(lambda: streamlit_ready(EVAL_PORT), 45, "评测看板")
        self.set_message("评测看板已就绪。")
        webbrowser.open(f"http://127.0.0.1:{EVAL_PORT}")

    def apply_health(self, health: dict) -> None:
        identity = health_identity(health)
        if identity in {"legacy_project", "old_project"}:
            self.set_status("api", "检测到旧版本服务")
        elif identity == "foreign":
            self.set_status("api", "端口被其他服务占用")
        else:
            self.set_status("api", "运行中" if health.get("api_ready") else "异常")
        knowledge = health.get("knowledge_base") or {}
        index = health.get("vector_index") or {}
        llm = health.get("llm") or {}
        self.set_status(
            "knowledge",
            f"{knowledge.get('status', 'unknown')} · {knowledge.get('document_count', '—')} PDFs",
        )
        self.set_status(
            "index",
            f"{index.get('status', 'unknown')} · {index.get('chunk_count', '—')} chunks",
        )
        embedding = health.get("embedding") or {}
        self.set_status(
            "embedding",
            f"{embedding.get('status', 'unknown')} · {embedding.get('model', '—')}",
        )
        self.set_status(
            "version",
            f"{health.get('build_id', 'legacy')} · {health.get('prompt_version', 'legacy')}",
        )
        self.set_status(
            "llm",
            f"{llm.get('status', 'unknown')} · {llm.get('model', '—')}",
        )

    def refresh_worker(self, show_message: bool = False) -> None:
        try:
            health = request_json(f"{API_BASE}/health", timeout=3)
            identity = health_identity(health)
            if identity in {"current", "legacy_project", "old_project"}:
                self.apply_health(health)
                if identity in {"legacy_project", "old_project"}:
                    self.set_message(
                        "检测到旧版本服务。点击“启动项目”将先验证并停止旧进程树，再启动当前版本。"
                    )
            elif port_open(API_PORT):
                self.set_status("api", f"端口冲突 ({API_PORT})")
            else:
                self.set_status("api", "未启动")
                pdf_count, index_exists, llm_configured, llm_model = offline_project_status()
                self.set_status("knowledge", f"{'ready' if pdf_count else 'missing'} · {pdf_count} PDFs")
                self.set_status("index", "文件存在 · 服务未检查" if index_exists else "missing")
                self.set_status("embedding", "服务未检查")
                self.set_status(
                    "version",
                    f"expected {EXPECTED_BUILD['build_id']}",
                )
                self.set_status(
                    "llm",
                    f"{'configured' if llm_configured else 'not_configured'} · {llm_model}",
                )

            ui_owned, _ = owned_process("ui")
            if streamlit_ready(UI_PORT) and ui_owned:
                self.set_status("ui", "运行中")
            elif port_open(UI_PORT):
                self.set_status("ui", f"端口冲突 ({UI_PORT})")
            else:
                self.set_status("ui", "未启动")
            if show_message:
                self.set_message("健康检查已完成。")
        finally:
            self.refresh_running = False

    def refresh_async(self, show_message: bool = False) -> None:
        if self.refresh_running:
            return
        self.refresh_running = True
        threading.Thread(target=self.refresh_worker, args=(show_message,), daemon=True).start()

    def periodic_refresh(self) -> None:
        self.refresh_async()
        self.root.after(15000, self.periodic_refresh)

    def show_knowledge_status(self) -> None:
        status = request_json(f"{API_BASE}/knowledge-base/status", timeout=5)
        if not status:
            messagebox.showwarning("知识库状态", "后端未启动，无法读取知识库状态。")
            return
        knowledge = status.get("knowledge_base") or {}
        index = status.get("vector_index") or {}
        embedding = status.get("embedding") or {}
        messagebox.showinfo(
            "知识库状态",
            "\n".join(
                [
                    f"知识库：{knowledge.get('status')} ({knowledge.get('document_count')} PDFs)",
                    f"向量库：{index.get('type')} / {index.get('status')}",
                    f"Chunks：{index.get('chunk_count')}",
                    f"索引更新时间：{index.get('last_updated') or 'unknown'}",
                    f"Embedding：{embedding.get('model')} / {embedding.get('status')}",
                    f"Build：{status.get('build_id')}",
                    f"Prompt：{status.get('prompt_version')}",
                    f"Prewarmed：{status.get('prewarmed')}",
                ]
            ),
        )

    def open_logs(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(LOG_DIR)


def acquire_single_instance() -> int | None:
    if sys.platform != "win32":
        return 1
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\LiteratureRAGLauncher")
    if not handle or ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        return None
    return handle


def main() -> None:
    mutex = acquire_single_instance()
    if mutex is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Literature RAG", "启动器已经打开。")
        root.destroy()
        return
    root = tk.Tk()
    Launcher(root)
    try:
        root.mainloop()
    finally:
        if sys.platform == "win32" and mutex not in (None, 1):
            ctypes.windll.kernel32.CloseHandle(mutex)


if __name__ == "__main__":
    main()
