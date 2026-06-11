use crate::{
    cli::ModelArch,
    models::training::{data_loader::DataLoader, model_architectures::PricePredictionModel},
};

pub mod data_loader;
mod model_architectures;

pub fn train(arch: &ModelArch, dataset_path: &str) {
    tracing_subscriber::fmt::init();

    let training_data = DataLoader::new(dataset_path)
        .load()
        .build_features(100)
        .expect("cannot create features");

    let mut model = PricePredictionModel::default();
    model.train(arch, &training_data).save("model.json");
}
