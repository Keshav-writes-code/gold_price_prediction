use std::{fs::File, path::PathBuf};

use linfa::traits::Fit;
use linfa_linear::{FittedLinearRegression, LinearRegression};
use ndarray::{Array1, Array2};

use crate::models::training::{data_loader::TrainingData, model_architectures::Modelable};

pub struct LR {
    model: FittedLinearRegression<f64>,
}

impl LR {
    pub fn train(data: &TrainingData) -> Self {
        let dataset: linfa::DatasetBase<Array2<f64>, Array1<f64>> =
            linfa::Dataset::new(data.x_train.clone(), data.y_train.clone());
        let model = LinearRegression::default()
            .fit(&dataset)
            .expect("Cannot train");

        Self { model }
    }
}
impl Modelable for LR {
    fn save(&self, save_path: &PathBuf) {
        let fd = File::create(save_path).unwrap();
        serde_json::to_writer(fd, &self.model).expect("Cannot create Model file");
    }
}
