import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


st.set_page_config(page_title='Wine Quality', page_icon='🍷', layout='wide')

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.joblib'
TEST_PATH = BASE_DIR / 'test.csv'
CONFIG_PATH = BASE_DIR / 'config.json'


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_test_data(path):
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model(MODEL_PATH)
test_data = load_test_data(TEST_PATH)
config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

feature_names = [column for column in test_data.columns if column != 'target']
X_test = test_data[feature_names]
y_test = test_data['target'].to_numpy()
test_scores = test_proba(model, X_test)

st.title('Оценка качества вина')

threshold = st.sidebar.slider(
    'Порог решения',
    min_value=0.0,
    max_value=1.0,
    value=float(config['threshold']),
    step=0.01
)
st.sidebar.caption('Вино считается хорошим, если вероятность не меньше порога.')

single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs([
    'Оценка одного вина',
    'Модельные метрики',
    'Пакетное предсказание',
    'Что выучила модель'
])

with single_tab:
    st.subheader('Характеристики вина')
    input_values = {}
    numeric_features = [name for name in feature_names if name != 'is_red']
    columns = st.columns(3)
    for index, feature in enumerate(numeric_features):
        values = X_test[feature]
        with columns[index % 3]:
            input_values[feature] = st.number_input(
                feature,
                min_value=float(values.min()),
                max_value=float(values.max()),
                value=float(values.median()),
                format='%.4f'
            )

    input_values['is_red'] = int(st.toggle('Красное вино', value=False))
    one_wine = pd.DataFrame([input_values], columns=feature_names)
    probability = float(model.predict_proba(one_wine)[0, 1])
    decision = probability >= threshold

    left, right = st.columns(2)
    left.metric('Вероятность хорошего вина', f'{probability:.1%}')
    right.metric('Решение', 'Хорошее' if decision else 'Не хорошее')

with metrics_tab:
    predictions = (test_scores >= threshold).astype(int)
    confusion = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)

    metric_columns = st.columns(3)
    metric_columns[0].metric('Precision', f'{precision:.3f}')
    metric_columns[1].metric('Recall', f'{recall:.3f}')
    metric_columns[2].metric('F1', f'{f1:.3f}')

    plot_columns = st.columns(3)
    with plot_columns[0]:
        fig, ax = plt.subplots(figsize=(5, 4))
        metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=['Не хорошее', 'Хорошее']
        ).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.grid(False)
        ax.set_title('Матрица ошибок')
        st.pyplot(fig)
        plt.close(fig)

    tn, fp, fn, tp = confusion.ravel()
    fpr_point = fp / (fp + tn) if fp + tn else 0.0
    tpr_point = tp / (tp + fn) if tp + fn else 0.0
    fpr, tpr, _ = metrics.roc_curve(y_test, test_scores)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_test, test_scores)

    with plot_columns[1]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, test_scores):.3f}')
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.scatter(fpr_point, tpr_point, color='crimson', zorder=3, label=f'Порог {threshold:.2f}')
        ax.set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with plot_columns[2]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(
            recall_curve,
            precision_curve,
            label=f'PR-AUC = {metrics.average_precision_score(y_test, test_scores):.3f}'
        )
        ax.axhline(y_test.mean(), ls='--', color='gray', label='Случайная модель')
        ax.scatter(recall, precision, color='crimson', zorder=3, label=f'Порог {threshold:.2f}')
        ax.set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        'Если увеличить порог, в подборке обычно станет меньше вин: precision вырастет, '
        'а recall снизится. При уменьшении порога подборка расширяется, recall растёт, '
        'но вместе с хорошими винами чаще попадаются посредственные.'
    )

with batch_tab:
    uploaded_file = st.file_uploader('Загрузите CSV с признаками вин', type='csv')
    if uploaded_file is not None:
        try:
            uploaded = pd.read_csv(uploaded_file)
            missing = [feature for feature in feature_names if feature not in uploaded.columns]
            if missing:
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            elif uploaded.empty:
                st.error('Файл не содержит строк с данными.')
            else:
                batch_features = uploaded[feature_names].apply(pd.to_numeric, errors='coerce')
                if batch_features.isna().any().any():
                    bad_columns = batch_features.columns[batch_features.isna().any()].tolist()
                    st.error(
                        'В признаках есть пропуски или нечисловые значения: '
                        + ', '.join(bad_columns)
                    )
                else:
                    batch_scores = model.predict_proba(batch_features)[:, 1]
                    result = uploaded.copy()
                    result['probability'] = batch_scores
                    result['prediction'] = (batch_scores >= threshold).astype(int)
                    st.dataframe(result, use_container_width=True)
                    st.download_button(
                        'Скачать предсказания',
                        data=result.to_csv(index=False).encode('utf-8'),
                        file_name='wine_predictions.csv',
                        mime='text/csv'
                    )
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f'Не удалось прочитать CSV: {error}')

with coefficients_tab:
    classifier = model.steps[-1][1]
    coefficients = pd.Series(classifier.coef_[0], index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(coefficients >= 0, 'seagreen', 'indianred')
    coefficients.plot.barh(ax=ax, color=colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Коэффициент после масштабирования признаков')
    ax.set_title('Вклад признаков в прогноз')
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        f'Сильнее всего за хорошее вино голосует **{coefficients.idxmax()}**, '
        f'а против — **{coefficients.idxmin()}**. Положительный коэффициент повышает '
        'оценку модели при росте признака, отрицательный — снижает.'
    )
