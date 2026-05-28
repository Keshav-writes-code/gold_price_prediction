use std::{
    fs::{File, create_dir_all},
    path::Path,
};

pub const ARTIFACTS_DIR: &str = ".artifacts";

pub fn init() {
    create_dir_all(ARTIFACTS_DIR).expect("cannot cerate dir");
}
pub fn create_artifact(sub_path: &str) -> File {
    let path = Path::new(ARTIFACTS_DIR).join(sub_path);
    if let Some(parent_path) = path.parent() {
        create_dir_all(parent_path).expect("cannot create parent dirs");
    }

    File::create(path).expect("cannot create path")
}
pub fn open_artifact(sub_path: &str) -> File {
    let path = Path::new(ARTIFACTS_DIR).join(sub_path);
    File::open(path).expect("cannot open file")
}
