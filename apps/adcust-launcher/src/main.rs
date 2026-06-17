#![windows_subsystem = "windows"]

use std::env;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

const CREATE_NO_WINDOW: u32 = 0x08000000;
const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: u16 = 8000;

#[link(name = "user32")]
extern "system" {
    fn MessageBoxW(hwnd: *mut core::ffi::c_void, text: *const u16, caption: *const u16, typ: u32) -> i32;
}

fn main() {
    if let Err(error) = run() {
        show_error("AdCust Launcher", &error);
    }
}

fn run() -> Result<(), String> {
    let base_dir = current_exe_dir()?;
    let frontend_exe = base_dir.join("AdCust.exe");
    let runtime_backend = base_dir.join("runtime").join("backend");
    let runtime_logic = base_dir.join("runtime").join("adcust-logic");
    let data_dir = base_dir.join("data");
    let logs_dir = base_dir.join("logs");

    ensure_exists(&frontend_exe, "AdCust.exe")?;
    ensure_exists(&runtime_backend, "backend runtime")?;
    ensure_exists(&runtime_logic, "adcust-logic runtime")?;
    fs::create_dir_all(&data_dir).map_err(|err| format!("Cannot create data directory: {err}"))?;
    fs::create_dir_all(&logs_dir).map_err(|err| format!("Cannot create logs directory: {err}"))?;

    if !backend_is_ready() {
        start_backend(&runtime_backend, &runtime_logic, &data_dir, &logs_dir)?;
        wait_for_backend()?;
    }

    Command::new(&frontend_exe)
        .current_dir(&base_dir)
        .spawn()
        .map_err(|err| format!("Cannot launch AdCust desktop app: {err}"))?;

    Ok(())
}

fn current_exe_dir() -> Result<PathBuf, String> {
    let exe = env::current_exe().map_err(|err| format!("Cannot locate launcher executable: {err}"))?;
    exe.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Cannot locate launcher directory.".to_string())
}

fn ensure_exists(path: &Path, label: &str) -> Result<(), String> {
    if path.exists() {
        Ok(())
    } else {
        Err(format!("Missing {label}: {}", path.display()))
    }
}

fn start_backend(
    runtime_backend: &Path,
    runtime_logic: &Path,
    data_dir: &Path,
    logs_dir: &Path,
) -> Result<(), String> {
    let python = env::var("ADCUST_RELEASE_PYTHON")
        .unwrap_or_else(|_| "D:\\runtime\\miniconda3\\envs\\heavy-common-env\\python.exe".to_string());
    if !Path::new(&python).exists() {
        return Err(format!(
            "Python executable not found: {python}\n\nSet ADCUST_RELEASE_PYTHON and rebuild the release if your Python path is different."
        ));
    }

    let backend_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(logs_dir.join("backend.log"))
        .map_err(|err| format!("Cannot open backend log file: {err}"))?;
    let backend_error_log = backend_log
        .try_clone()
        .map_err(|err| format!("Cannot clone backend log file: {err}"))?;

    let python_path = format!("{};{}", runtime_backend.display(), runtime_logic.display());

    Command::new(python)
        .arg("-m")
        .arg("uvicorn")
        .arg("app.main:app")
        .arg("--app-dir")
        .arg(runtime_backend)
        .arg("--host")
        .arg(BACKEND_HOST)
        .arg("--port")
        .arg(BACKEND_PORT.to_string())
        .env("ADCUST_RELEASE_ROOT", runtime_backend.parent().and_then(Path::parent).unwrap_or(runtime_backend))
        .env("ADCUST_BACKEND_HOST", BACKEND_HOST)
        .env("ADCUST_BACKEND_PORT", BACKEND_PORT.to_string())
        .env("ADCUST_DATA_DIR", data_dir)
        .env("PYTHONPATH", python_path)
        .stdout(Stdio::from(backend_log))
        .stderr(Stdio::from(backend_error_log))
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map_err(|err| format!("Cannot start AdCust backend: {err}"))?;

    Ok(())
}

fn wait_for_backend() -> Result<(), String> {
    for _ in 0..40 {
        if backend_is_ready() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err("Backend did not become ready on http://127.0.0.1:8000/health. Check logs/backend.log.".to_string())
}

fn backend_is_ready() -> bool {
    let address: SocketAddr = match format!("{BACKEND_HOST}:{BACKEND_PORT}").parse() {
        Ok(address) => address,
        Err(_) => return false,
    };
    let mut stream = match TcpStream::connect_timeout(&address, Duration::from_millis(500)) {
        Ok(stream) => stream,
        Err(_) => return false,
    };

    let _ = stream.set_read_timeout(Some(Duration::from_millis(800)));
    let request = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }

    let mut response = String::new();
    if stream.read_to_string(&mut response).is_err() {
        return false;
    }

    response.contains("200 OK") && response.contains("adcust-backend") && response.contains("ready")
}

fn show_error(title: &str, message: &str) {
    let title = to_wide(title);
    let message = to_wide(message);
    unsafe {
        MessageBoxW(core::ptr::null_mut(), message.as_ptr(), title.as_ptr(), 0x00000010);
    }
}

fn to_wide(value: &str) -> Vec<u16> {
    OsStr::new(value).encode_wide().chain(Some(0)).collect()
}
