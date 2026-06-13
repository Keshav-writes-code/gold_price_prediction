use actix_cors::Cors;
use actix_web::{App, HttpResponse, HttpServer, Responder, get, post, web};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{config::ModelArch, models::infrence::PricePredictionModelInfrence};

#[derive(Deserialize)]
struct PredictionReq {
    pub target_time_unix: Vec<i64>,
}

#[derive(Serialize)]
struct PredictionResponse {
    pub predicted_price: Vec<f64>,
}

#[actix_web::main]
pub async fn serve(arch: &ModelArch, dataset_path: &str) {
    tracing_subscriber::fmt::init();

    let model = PricePredictionModelInfrence::new(arch, dataset_path);
    let model_arc = web::Data::new(model);

    info!("Serving the WebApp at http://0.0.0.0:8080/");
    info!("Serving the prediction at http://0.0.0.0:8080/predict");
    HttpServer::new(move || {
        let cors = Cors::permissive();
        App::new()
            .app_data(model_arc.clone())
            .wrap(cors)
            .service(predict)
            .service(index)
    })
    .bind(("0.0.0.0", 8080))
    .expect("cannot connect to Socket")
    .run()
    .await
    .expect("cannot run server");
}

const DASHBOARD_HTML: &str = include_str!("./static/index.html");

#[get("/")]
async fn index() -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(DASHBOARD_HTML)
}

#[post("/predict")]
async fn predict(
    model: web::Data<PricePredictionModelInfrence>,
    req: web::Json<PredictionReq>,
) -> impl Responder {
    let req_data = req.into_inner();

    info!("Serving the prediction at http://0.0.0.0:8080/predict");
    let mut predictions = Vec::new();
    for input in req_data.target_time_unix {
        predictions.push(model.predict(input));
    }
    HttpResponse::Ok().json(PredictionResponse {
        predicted_price: predictions,
    })
}
