use argh::FromArgs;

#[derive(FromArgs)]
/// Inpouts.
struct Input {
    /// start the taining loop and output a model artifact
    #[argh(switch, short = 't')]
    train: bool,

    /// visualize the data
    #[argh(switch, short = 'v')]
    viz: bool,

    /// pull_and_store_data
    #[argh(switch, short = 'p')]
    pull: bool,
}

fn main() {
    let inputs: Input = argh::from_env();
    if inputs.pull {
        gold_price_prediction::injestion::hanlde_injestion();

        // logic to download data from cagle and store then in a sqlite database
    }
    if inputs.viz {
        // logic to visualize data stored in sqlite
    }
    if inputs.train {}
}
