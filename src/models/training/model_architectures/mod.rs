use std::path::PathBuf;

use crate::{
    cli::ModelArch,
    config::create_artifact,
    models::training::{
        data_loader::TrainingData,
        model_architectures::{linear_regression::LR, mlp::MLP},
    },
};

mod linear_regression;
mod mlp;

pub trait Modelable {
    fn train(&mut self, data: &TrainingData);
    fn save(&self, save_path: &PathBuf);
}

enum Models {
    LinearRegression(LR),
    Mlp(MLP),
}

impl Modelable for Models {
    fn train(&mut self, data: &TrainingData) {
        match self {
            Models::LinearRegression(model) => model.train(data),
            Models::Mlp(model) => model.train(data),
        }
    }
    fn save(&self, save_path: &PathBuf) {
        match self {
            Models::LinearRegression(model) => model.save(save_path),
            Models::Mlp(model) => model.save(save_path),
        }
    }
}

#[derive(Default)]
pub struct PricePredictionModel {
    model: Option<Models>,
}
impl PricePredictionModel {
    pub fn train(&mut self, arch: &ModelArch, training_data: &TrainingData) -> &Self {
        let mut model: Models = match arch {
            ModelArch::LinearRegression => Models::LinearRegression(LR::default()),
            ModelArch::Mlp => Models::Mlp(MLP::default()),
        };
        model.train(training_data);
        self.model = Some(model);
        self
    }
    pub fn save(&self, path: &str) {
        let path = create_artifact(path);
        self.model.as_ref().expect("Cannot find model").save(&path);
    }
}
