use argh::FromArgs;

use crate::config::get_config;

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

    /// config file path
    #[argh(option, short = 'c')]
    config_file: Option<String>,
}

pub fn start_cli() {
    crate::utility::init();
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
        let (arch, lr) = get_config(inputs.config_file.as_deref());
        crate::models::training::train(&arch, lr, "raw_data.csv");
    }
    if inputs.serve {
        let (arch, _) = get_config(inputs.config_file.as_deref());
        crate::serving::serve(&arch, "raw_data.csv");
    }
}
