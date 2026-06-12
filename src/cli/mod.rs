use argh::{FromArgValue, FromArgs};

#[derive(FromArgValue, Clone, Copy)]
pub enum ModelArch {
    LinearRegression,
    Mlp,
}

#[derive(FromArgs)]
/// Inpouts.
struct Input {
    /// start the taining loop and output a model artifact
    #[argh(switch, short = 't')]
    train: bool,

    #[cfg(feature = "visualization")]
    /// visualize the data
    #[argh(switch, short = 'v')]
    viz: bool,

    /// pull_and_store_data
    #[argh(switch, short = 'p')]
    pull: bool,

    /// serve the model
    #[argh(switch, short = 's')]
    serve: bool,

    /// the model architecture to use for traning or infrence ('linear_regression', 'mlp')
    #[argh(option, short = 'a', default = "ModelArch::LinearRegression")]
    arch: ModelArch,

    /// the learning rate while trnaing
    #[argh(option, short = 'l', default = "0.04")]
    lr: f32,
}

pub fn start_cli() {
    crate::config::init();
    let inputs: Input = argh::from_env();
    if inputs.pull {
        // logic to download data from cagle and store then in a sqlite database
        crate::data::ingestion::hanlde_ingestion("raw_data.csv");
    }
    #[cfg(feature = "visualization")]
    if inputs.viz {
        // logic to visualize data
        crate::data::visulization::init_visulization();
    }
    if inputs.train {
        crate::models::training::train(&inputs.arch, inputs.lr, "raw_data.csv");
    }
    if inputs.serve {
        crate::serving::serve(&inputs.arch, "raw_data.csv");
    }
}
