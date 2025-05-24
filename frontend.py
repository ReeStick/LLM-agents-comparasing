import streamlit as st
import requests

# Заголовок
st.title("LLM Evaluation Interface")

# Выбор датасета и метода
dataset = st.selectbox("Выберите датасет", ["squad_v2", "ag_news", "snli", "trec", "wmt14"])
method = st.selectbox("Выберите метод", ["rag", "graph_rag", "pln"])

# Кнопка запуска
if st.button("Запустить оценку"):
    with st.spinner("Оцениваем..."):
        try:
            url = "http://127.0.0.1:9888/evaluate"
            payload = {
                "dataset": "squad_v2",
                "method": "rag"
            }

            response = requests.post(url, json=payload)

            print(f"Status Code: {response.status_code}")
            print("Response JSON:", response.json())
            if response.status_code == 200:
                result = response.json()
                st.success("Готово!")
                st.write("Результаты метрик:")
                for metric, value in result['result'].items():
                    st.write(f"**{metric}**: {float(value):.4f}")
            else:
                st.error(f"Ошибка: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Не удалось подключиться к серверу: {e}")
