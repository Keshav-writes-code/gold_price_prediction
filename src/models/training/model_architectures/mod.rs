use crate::{
    cli::ModelArch,
    models::training::{
        data_loader::TrainingData,
        model_architectures::{linear_regression::LR, mlp::MLP},
    },
};

mod linear_regression;
mod mlp;

pub trait Modelable: Sync + Send {
    fn train(&mut self, data: &TrainingData);
    fn save(&self);
    fn load(&mut self);
    fn predict(&self, x_input: &[f64]) -> f64;
}

pub struct PricePredictionModel {
    model: Option<Box<dyn Modelable>>,
}
impl PricePredictionModel {
    pub fn new(arch: &ModelArch) -> Self {
        let model: Box<dyn Modelable> = match arch {
            ModelArch::LinearRegression => Box::new(LR::default()),
            ModelArch::Mlp => Box::new(MLP::default()),
        };
        Self { model: Some(model) }
    }
    pub fn train(&mut self, training_data: &TrainingData) -> &Self {
        self.model
            .as_mut()
            .expect("Model must of initialized bwfore calling train()")
            .train(training_data);
        self
    }
    pub fn save(&self) {
        self.model.as_ref().expect("Cannot find model").save();
    }
    pub fn load(&mut self) {
        self.model
            .as_mut()
            .expect("Model must be initialized before calling load()")
            .load();
    }
    pub fn predict(&self, x_input: &[f64]) -> f64 {
        self.model
            .as_ref()
            .expect("Model must be initialized before calling load()")
            .predict(x_input)
    }
}
