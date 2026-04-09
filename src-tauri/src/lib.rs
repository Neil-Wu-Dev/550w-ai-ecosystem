use serde::{Deserialize, Serialize};

// --- 1. 数据结构 ---
#[derive(Deserialize)]
struct ModelSelectRequest {
    local_path: String,
}

#[derive(Serialize)]
struct ModelResponse {
    name: String,
    local_path: String,
    architecture: String,
    precision: String,
    trainable_layers: Vec<String>,
}

// --- 2. 功能命令 ---
#[tauri::command]
fn get_options() -> serde_json::Value {
    serde_json::json!({
        "strategies": ["smoke", "knowledge", "behavior"],
        "modes": ["sequel", "blended"]
    })
}

#[tauri::command]
fn select_model(req: ModelSelectRequest) -> Result<ModelResponse, String> {
    if req.local_path.is_empty() { 
        return Err("PATH_EMPTY: Please select a valid model path.".into()); 
    }
    Ok(ModelResponse {
        name: "Llama-3-Adcust-Base".into(),
        local_path: req.local_path,
        architecture: "Transformer / Llama".into(),
        precision: "BF16".into(),
        trainable_layers: vec!["q_proj".into(), "v_proj".into()],
    })
}

#[tauri::command]
fn start_train(
    target_dir: String, 
    layers: Vec<String>, 
    mode: String, 
    name: Option<String>
) -> Result<String, String> {
    // 模拟耗时操作
    std::thread::sleep(std::time::Duration::from_secs(2));
    
    let model_name = name.unwrap_or_else(|| "model".into());
    Ok(format!("{}/{}_final.bin", target_dir, model_name))
}

// --- 3. 运行配置 ---
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init()) 
        .invoke_handler(tauri::generate_handler![
            get_options, 
            select_model,
            start_train
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}