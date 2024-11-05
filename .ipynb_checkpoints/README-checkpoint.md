# Yandex Cup ML Challenge '24
## RecSys Track

[**Задача**](https://yandex.ru/cup/ml/): Определение кавер версий трека по акустическим признакам. Каждый трек описывается с помощью CQT-спектрограммы сжатой по размерности времени, которую строили по 60 секундам взятым из центральной части трека.

**Public nDCG**: *0.64461* \
**Private nDCG**: *0.63515*

To reproduce the results:
1. `conda create -n ya python=3.10`, `source activate ya`, `pip install -r requirements.txt`
2. Run script `run_train_models.sh`
3. Add the paths for the models from step 2 to the `paths` variable in file `ensemble_final.py`. Ensemble results using script `run_ensemble.sh`