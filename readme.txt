run air_quality_model_create.py and health_model_create.py to generate .pkl files.
install fastAPI
then run mainAPI.py

api call sample:


POST http://127.0.0.1:8000/predictaqi HTTP/1.1
content-type: application/json

{
  "city": "Mumbai",
  "date": "2026-11-10"
}


POST http://127.0.0.1:8000/predicthealth HTTP/1.1
content-type: application/json

{"AQI": 75}
