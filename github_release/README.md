# Система виявлення аномалій ГЦН-195М

Проєкт містить:

- Xcos-модель ГЦН-195М;
- Python-пайплайн підготовки даних і навчання `Random Forest`;
- модулі inference, explanation і short-term forecast.

## Структура

```text
github_release/
  xcos/
  python_ml/
  python_ml/knowledge_base/
  data_demo/
  models/
  figures/
```

## Вимоги

- Scilab/Xcos
- Python 3.10+

Встановлення залежностей:

```bash
python -m pip install -r requirements.txt
```

## Запуск Xcos

У Scilab:

```scilab
cd("xcos");
exec("run_gcn195m_model.sce", -1);
```

Після запуску CSV-файли режимів будуть у `data_demo/`.

## Запуск Python-пайплайна

З кореня репозиторію:

```bash
python python_ml/run_section34_pipeline.py
```

Скрипт:

- читає CSV із `data_demo/`;
- формує об’єднаний і віконний датасети;
- навчає модель;
- зберігає результати у `generated/`.

## Inference

```bash
python python_ml/predict_gcn195m_mode.py --csv data_demo/gcn_normal.csv
```

## Пояснення і прогноз

```bash
python python_ml/explain_and_forecast_gcn195m.py --csv data_demo/gcn_motor_overload.csv
```

Результати explanation і forecast також записуються у `generated/`.
