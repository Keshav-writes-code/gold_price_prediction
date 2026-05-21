use std::fs::File;

use polars::prelude::{UnionArgs, concat};
use polars::{io::parquet::write::ParquetWriter, lazy::frame::LazyFrame};

use self::internet_zip_injestion::ZipInjestion;

mod api_injestion;
mod internet_zip_injestion;

pub trait InjestionStratergy {
    fn fetch_data(&self) -> LazyFrame;
}
pub const FEATURE_STORE_PATH: &str = "output.parquet";
pub fn hanlde_injestion() {
    let stratergies: Vec<Box<dyn InjestionStratergy>> = vec![Box::new(ZipInjestion)];

    let mut frames = Vec::new();
    for stratergy in stratergies {
        frames.push(stratergy.fetch_data());
    }

    if !frames.is_empty() {
        let combined_lf = concat(frames, UnionArgs::default()).unwrap();
        let mut df = combined_lf.collect().unwrap();

        let mut file = File::create(&FEATURE_STORE_PATH).unwrap();
        ParquetWriter::new(&mut file)
            .with_compression(polars::io::parquet::write::ParquetCompression::Zstd(None))
            .finish(&mut df)
            .unwrap();
    }
}
