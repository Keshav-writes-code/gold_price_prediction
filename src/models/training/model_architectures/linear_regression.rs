use linfa::traits::{Fit, Predict};
use linfa_linear::{FittedLinearRegression, LinearRegression};
use ndarray::{Array1, Array2, ArrayView};

use crate::{
    config::{create_artifact, open_artifact},
    models::training::{data_loader::TrainingData, model_architectures::Modelable},
};

pub struct LR {
    model: Option<FittedLinearRegression<f64>>,
    model_file_name: String,
}

impl Default for LR {
    fn default() -> Self {
        Self {
            model: None,
            model_file_name: "lr_model.json".to_string(),
        }
    }
}

impl Modelable for LR {
    fn load(&mut self) {
        let fd = open_artifact(&self.model_file_name);
        let model: Option<FittedLinearRegression<f64>> =
            serde_json::from_reader(&fd).expect("Cannot Desearlize");
        self.model = model
    }
    fn save(&self) {
        let fd = create_artifact(&self.model_file_name);
        serde_json::to_writer(fd, &self.model).expect("Cannot create Model file");
    }
    fn train(&mut self, data: &TrainingData, _lr: f32) {
        let dataset: linfa::DatasetBase<Array2<f64>, Array1<f64>> =
            linfa::Dataset::new(data.x_train.clone(), data.y_train.clone());
        let model = LinearRegression::default()
            .fit(&dataset)
            .expect("Cannot train");
        self.model = Some(model);
    }
    fn predict(&self, x_input: &[f64]) -> f64 {
        let model = self
            .model
            .as_ref()
            .expect("You cannot call .predict before calling .load or .new");

        let features =
            ArrayView::from_shape((1, x_input.len()), x_input).expect("Cammpt create a arrau voew");

        model.predict(features).targets()[0]
    }
}
