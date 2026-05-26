use std::fs::File;

use actix_web::{App, HttpResponse, HttpServer, Responder, get, web};
use linfa::traits::Predict;
use linfa_linear::FittedLinearRegression;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct PredictionReq {
    pub features: Vec<f64>,
}

#[derive(Serialize)]
struct PredictionResponse {
    pub predicted_price: f64,
}

#[actix_web::main]
pub async fn serve() {
    let file = File::open("gold_price_prediction.json").expect("Cannot Open File");
    let model: FittedLinearRegression<f64> =
        serde_json::from_reader(file).expect("cannot load model");

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(model.clone()))
            .service(predict)
    })
    .bind(("0.0.0.0", 8080))
    .expect("cannot connect to Socket")
    .run()
    .await
    .expect("cannot run server");
}

#[get("/predict")]
async fn predict(
    model: web::Data<FittedLinearRegression<f64>>,
    req: web::Json<PredictionReq>,
) -> impl Responder {
    let req_data = req.into_inner();
    let input = Array2::from_shape_vec((1, 99), req_data.features)
        .expect("Feature vectors must contain 99 elements");

    let pred = model.predict(&input);
    HttpResponse::Ok().json(PredictionResponse {
        predicted_price: pred[0],
    })
}
