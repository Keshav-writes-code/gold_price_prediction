use actix_cors::Cors;
use actix_web::{App, HttpResponse, HttpServer, Responder, post, web};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::models::infrence::PricePredictionModelInfrence;

#[derive(Deserialize)]
struct PredictionReq {
    pub target_time_unix: i64,
}

#[derive(Serialize)]
struct PredictionResponse {
    pub predicted_price: f64,
}

#[actix_web::main]
pub async fn serve() {
    tracing_subscriber::fmt::init();
    let model = PricePredictionModelInfrence::default();
    let model_arc = web::Data::new(model);

    info!("Serving the prediction at http://0.0.0.0:8080/predict");
    HttpServer::new(move || {
        let cors = Cors::permissive();
        App::new()
            .app_data(model_arc.clone())
            .wrap(cors)
            .service(predict)
    })
    .bind(("0.0.0.0", 8080))
    .expect("cannot connect to Socket")
    .run()
    .await
    .expect("cannot run server");
}

#[post("/predict")]
async fn predict(
    model: web::Data<PricePredictionModelInfrence>,
    req: web::Json<PredictionReq>,
) -> impl Responder {
    let req_data = req.into_inner();

    info!("Serving the prediction at http://0.0.0.0:8080/predict");
    let pred = model.predict(req_data.target_time_unix);
    HttpResponse::Ok().json(PredictionResponse {
        predicted_price: pred,
    })
}
