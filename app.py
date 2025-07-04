import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from AdvancedLSTM import AdvancedLSTMPredictor  # 假设你把核心类放入这个模块
import matplotlib.font_manager as fm
import os

font_path = os.path.join("fonts", "NotoSansCJKsc-Regular.otf")

if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    # 设置全局默认字体
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
    print(f"✅ 使用字体: {font_prop.get_name()}")
else:
    print("⚠️ 字体文件未找到，中文可能乱码")

plt.rcParams['axes.unicode_minus'] = False




def main():
    st.title("📈 高级双向LSTM + 注意力机制客流预测系统")
    st.write("上传包含 `日期`, `顾客数`, `假日` 等字段的 Excel 文件进行预测。")

    uploaded_file = st.file_uploader("上传Excel文件（例如：三福.xlsx）", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"数据加载成功！共 {df.shape[0]} 行，{df.shape[1]} 列")
            st.dataframe(df.head())

            # 模型初始化
            predictor = AdvancedLSTMPredictor(sequence_length=30)

            # 特征工程
            df = predictor.create_advanced_features(df)

            # 数据准备
            X, y, feature_cols = predictor.prepare_data(df)

            # 训练模型
            with st.spinner("模型训练中，请稍等..."):
                history = predictor.train_model(X, y, epochs=50, batch_size=32)

            st.success("✅ 模型训练完成！")

            # 未来预测
            future_days = st.slider("选择要预测的未来天数", min_value=1, max_value=30, value=7)
            future_predictions = predictor.predict_future_advanced(df, feature_cols, days=future_days)

            st.subheader("📅 未来预测结果")
            st.dataframe(future_predictions)

            # 绘图展示
            st.subheader("📊 预测结果可视化")
            fig, ax = plt.subplots(figsize=(14, 6))

            hist_days = 60
            hist_data = df.tail(hist_days)
            ax.plot(hist_data['日期'], hist_data['顾客数'], 'o-', label='历史客流', markersize=4)
            ax.plot(future_predictions['日期'], future_predictions['预测客流'], 
                    's-', label='预测客流', color='red', markersize=6, linewidth=2)
            ax.fill_between(future_predictions['日期'], 
                            future_predictions['预测下限'], 
                            future_predictions['预测上限'],
                            alpha=0.3, color='red', label='95%置信区间')
            ax.set_title('预测客流趋势', fontsize=16)
            ax.set_xlabel('日期')
            ax.set_ylabel('客流量')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # 下载链接
            csv = future_predictions.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 下载预测结果", data=csv, file_name='lstm_predictions.csv')

        except Exception as e:
            st.error(f"❌ 数据处理出错: {e}")

if __name__ == '__main__':
    main()
