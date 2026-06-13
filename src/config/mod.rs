use std::fs;

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
    if let Ok(_) = fs::exists(CONFIG_FILE_NAME) {
        let str_data =
            fs::read_to_string("train_config.toml").expect("cannot find traning config file");
        let data: Config = toml::from_str(&str_data).unwrap_or_default();
        (data.arch, data.learning_rate)
    } else {
        let config = Config::default();
        let content_toml = toml::to_string(&config).expect("cannot serialize data");
        fs::write(CONFIG_FILE_NAME, content_toml);
        (config.arch, config.learning_rate)
    }
}
