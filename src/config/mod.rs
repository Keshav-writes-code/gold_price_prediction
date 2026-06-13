use std::{fs, io::ErrorKind};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub enum ModelArch {
    LinearRegression,
    Mlp,
}

#[derive(Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    arch: ModelArch,
    learning_rate: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            arch: ModelArch::LinearRegression,
            learning_rate: 0.016,
        }
    }
}

pub fn get_config(file_path: Option<&str>) -> (ModelArch, f32) {
    const DEFAULT_PATH: &str = "training_config.toml";

    let (path, create_if_missing) = match file_path {
        Some(path) => (path, false),
        None => (DEFAULT_PATH, true),
    };

    let config = match fs::read_to_string(path) {
        Ok(content) => toml::from_str(&content).expect("Cammpt parse toml"),
        Err(e) if e.kind() == ErrorKind::NotFound && create_if_missing => {
            let default_config = Config::default();
            let content_toml = toml::to_string_pretty(&default_config).expect("Canot read toml");
            fs::write(path, &content_toml).expect("Cannot pase data");
            default_config
        }
        Err(_) => {
            panic!("Coudn't open file");
        }
    };

    (config.arch, config.learning_rate)
}
