use crate::{
    cli::ModelArch,
    models::training::{data_loader::DataLoader, model_architectures::PricePredictionModel},
};

pub mod data_loader;
pub mod model_architectures;

pub fn train(arch: &ModelArch, dataset_path: &str) {
    tracing_subscriber::fmt::init();

    let training_data = DataLoader::new(dataset_path)
        .load()
        .build_features(100)
        .expect("cannot create features");

    PricePredictionModel::new(arch).train(&training_data).save();
}
