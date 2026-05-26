use argh::FromArgs;

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
}

fn main() {
    let inputs: Input = argh::from_env();
    if inputs.pull {
        // logic to download data from cagle and store then in a sqlite database
        gold_price_prediction::data::ingestion::hanlde_ingestion();
    }
    #[cfg(feature = "visualization")]
    if inputs.viz {
        // logic to visualize data
        gold_price_prediction::data::visulization::init_visulization();
    }
    if inputs.train {
        gold_price_prediction::models::train();
    }
    if inputs.serve {
        gold_price_prediction::serving::serve();
    }
}
