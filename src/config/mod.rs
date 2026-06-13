use std::{fs, path::Path};

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

pub fn get_config() -> (ModelArch, f32) {
    const CONFIG_FILE_NAME: &str = "training_config.toml";
    if Path::new(CONFIG_FILE_NAME).exists() {
        let str_data =
            fs::read_to_string(CONFIG_FILE_NAME).expect("cannot find traning config file");
        let data: Config = toml::from_str(&str_data).unwrap_or_default();
        (data.arch, data.learning_rate)
    } else {
        let config = Config::default();
        let content_toml = toml::to_string(&config).expect("cannot serialize data");
        fs::write(CONFIG_FILE_NAME, content_toml).expect("Cannot create file");
        (config.arch, config.learning_rate)
    }
}
