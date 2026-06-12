use ndarray::Array1;
use rust_mlp::{Dataset, FitConfig, Metric, Mlp, MlpBuilder};

use crate::{
    config::{create_artifact_path, open_artifact_path},
    models::training::{data_loader::TrainingData, model_architectures::Modelable},
};

pub struct MLP {
    model: Option<Mlp>,
    model_file_name: String,
}

impl Default for MLP {
    fn default() -> Self {
        Self {
            model: None,
            model_file_name: "mlp_model.json".to_string(),
        }
    }
}

impl Modelable for MLP {
    fn train(&mut self, data: &TrainingData) {
        let x_train: Vec<Vec<f32>> = data
            .x_train
            .outer_iter()
            .map(|row| row.iter().map(|&x| x as f32).collect())
            .collect();

        let y_train: Vec<Vec<f32>> = data.y_train.iter().map(|&y| vec![y as f32]).collect();

        let dataset = rust_mlp::Dataset::from_rows(&x_train, &y_train).expect("Cannot get dataset");
        let mut mlp = MlpBuilder::new(data.x_dim)
            .unwrap()
            .add_layer(100, rust_mlp::Activation::ReLU)
            .unwrap()
            .add_layer(100, rust_mlp::Activation::ReLU)
            .unwrap()
            .add_layer(100, rust_mlp::Activation::ReLU)
            .unwrap()
            .build_with_seed(0)
            .unwrap();
        mlp.fit(
            &dataset,
            None,
            FitConfig {
                epochs: 100,
                lr: 0.2,
                batch_size: 4,
                shuffle: rust_mlp::Shuffle::Seeded(0),
                lr_schedule: rust_mlp::LrSchedule::Constant,
                optimizer: rust_mlp::Optimizer::Adam {
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                },
                weight_decay: 0.0,
                grad_clip_norm: None,
                loss: rust_mlp::Loss::Mse,
                metrics: vec![Metric::Accuracy],
            },
        )
        .expect("Cannot train model");
        self.model = Some(mlp);
    }
    fn save(&self) {
        let save_path = create_artifact_path(&self.model_file_name);
        self.model
            .as_ref()
            .expect("Called save on a model that doesn't have a model yet")
            .save_json(save_path);
    }
    fn load(&mut self) {
        let load_path = open_artifact_path(&self.model_file_name);
        self.model = Some(rust_mlp::Mlp::load_json(load_path).expect("cannot load MLP Model"));
    }
    fn predict(&self, x_input: &[f64]) -> f64 {
        let model = self
            .model
            .as_ref()
            .expect("Canot call .predict() beofre calling .new() or .load()");

        let vec_f32: Vec<f32> = x_input.iter().map(|&v| v as f32).collect();
        let mut output = vec![0.0_f32; model.output_dim()];
        let mut scratch = model.scratch();
        model.predict_into(&vec_f32, &mut scratch, &mut output);
        output[0] as f64
    }
}
