use std::{fs::File, path::PathBuf};

use rust_mlp::{FitConfig, Metric, Mlp, MlpBuilder};

use crate::models::training::{data_loader::TrainingData, model_architectures::Modelable};

pub struct MLP {
    model: Mlp,
}
impl MLP {
    pub fn train(data: &TrainingData) -> Self {
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
        Self { model: mlp }
    }
}
impl Modelable for MLP {
    fn save(&self, save_path: &PathBuf) {
        self.model.save_json(save_path);
    }
}
