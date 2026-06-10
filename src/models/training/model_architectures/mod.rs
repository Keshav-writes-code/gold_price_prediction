mod linear_regression;
mod mlp;

use std::path::PathBuf;

use crate::{
    cli::ModelArch,
    config::create_artifact,
    models::training::{
        data_loader::TrainingData,
        model_architectures::{linear_regression::LR, mlp::MLP},
    },
};

pub trait Modelable {
    fn save(&self, save_path: &PathBuf);
}

pub struct PricePredictionModel {
    trained_model: Box<dyn Modelable>,
}

impl PricePredictionModel {
    pub fn train(arch: &ModelArch, data: &TrainingData) -> Self {
        Self {
            trained_model: match arch {
                ModelArch::LinearRegression => Box::new(LR::train(data)),
                ModelArch::Mlp => Box::new(MLP::train(data)),
            },
        }
    }
    pub fn save(&self, path: &str) {
        let model_path = create_artifact(path);
        self.trained_model.save(&model_path);
    }
}
