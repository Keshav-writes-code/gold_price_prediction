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
}

fn main() {
    let inputs: Input = argh::from_env();
    if inputs.train {}
    if inputs.viz {}
}
