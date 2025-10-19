run : pip install -r requirements.txt


Train Hospital Surge Model: 

cd chinmay_mumabaihacks
python main.py
cd ..

Run health model and air quality model:

cd health_model
python health_model_create.py
cd ..

cd air_quality_model
python air_quality_model_create.py
cd ..

run hospital_api.py

python hospital_api.py

