import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

# 深度学习
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 数据处理
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib as mpl

# 1️⃣ 确保 SimHei.ttf 文件在你的仓库（比如放在根目录下 fonts/SimHei.ttf）
mpl.font_manager.fontManager.addfont("fonts/simhei.ttf")  # 注册字体
plt.rcParams['font.sans-serif'] = ['SimHei']   # 使用中文字体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# 设置页面配置
st.set_page_config(
    page_title="客流预测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# 注意力层定义
class AttentionLayer(layers.Layer):
    """自定义注意力层"""
    def __init__(self, units=32):
        super(AttentionLayer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='attention_weight'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_bias'
        )
        self.u = self.add_weight(
            shape=(self.units, 1),
            initializer='random_normal',
            trainable=True,
            name='attention_score'
        )
        
    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.u), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# 缓存模型类
@st.cache_resource
class LSTMPredictor:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.feature_cols = None
        
    def create_features(self, df):
        """创建特征"""
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        
        # 时间特征
        df['年'] = df['日期'].dt.year
        df['月'] = df['日期'].dt.month
        df['日'] = df['日期'].dt.day
        df['星期'] = df['日期'].dt.dayofweek
        df['季度'] = df['日期'].dt.quarter
        
        # 周期性编码
        df['年度进度'] = df['日期'].dt.dayofyear / 365
        df['年度_sin'] = np.sin(2 * np.pi * df['年度进度'])
        df['年度_cos'] = np.cos(2 * np.pi * df['年度进度'])
        df['月_sin'] = np.sin(2 * np.pi * df['月'] / 12)
        df['月_cos'] = np.cos(2 * np.pi * df['月'] / 12)
        df['星期_sin'] = np.sin(2 * np.pi * df['星期'] / 7)
        df['星期_cos'] = np.cos(2 * np.pi * df['星期'] / 7)
        df['日_sin'] = np.sin(2 * np.pi * df['日'] / 30)
        df['日_cos'] = np.cos(2 * np.pi * df['日'] / 30)
        
        # 特殊日期
        df['是否周末'] = (df['星期'] >= 5).astype(int)
        df['是否月初'] = (df['日'] <= 5).astype(int)
        df['是否月末'] = (df['日'] >= 25).astype(int)
        
        # 假日编码
        holiday_map = {'工作日': 0, '周末': 1, '节假日': 2}
        df['假日类型'] = df['假日'].map(holiday_map)
        
        # 移动统计
        for window in [7, 14, 30]:
            df[f'MA{window}'] = df['顾客数'].rolling(window=window, min_periods=1).mean()
            
        for window in [7, 14]:
            df[f'STD{window}'] = df['顾客数'].rolling(window=window, min_periods=1).std().fillna(0)
            
        # 趋势特征
        df['趋势7'] = df['MA7'] - df['MA14']
        df['趋势14'] = df['MA14'] - df['MA30']
        
        df = df.dropna()
        return df
    
    def build_model(self, input_shape):
        """构建模型"""
        inputs = Input(shape=input_shape)
        
        # 双向LSTM层
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.1)
        )(inputs)
        x = layers.BatchNormalization()(x)
        
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.1)
        )(x)
        x = layers.BatchNormalization()(x)
        
        # 注意力层
        attention_output, _ = AttentionLayer(32)(x)
        
        # 最后一层LSTM
        lstm_output = layers.Bidirectional(
            layers.LSTM(32, return_sequences=False)
        )(x)
        
        # 连接
        combined = layers.concatenate([attention_output, lstm_output])
        
        # 全连接层
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        
        return model

# 初始化session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictor' not in st.session_state:
    st.session_state.predictor = LSTMPredictor()
if 'df' not in st.session_state:
    st.session_state.df = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# 标题
st.markdown("<h1>🚀 客流预测</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>双向LSTM</p>", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("## 📋 控制面板")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传数据文件",
        type=['xlsx', 'xls', 'csv'],
        help="请上传包含客流数据的Excel或CSV文件"
    )
    
    st.markdown("---")
    
    # 模型参数
    st.markdown("### ⚙️ 模型参数")
    sequence_length = st.slider(
        "历史序列长度",
        min_value=7,
        max_value=60,
        value=30,
        help="使用多少天的历史数据进行预测"
    )
    
    epochs = st.slider(
        "训练轮数",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )
    
    batch_size = st.select_slider(
        "批次大小",
        options=[16, 32, 64, 128],
        value=32
    )
    
    st.markdown("---")
    
    # 预测参数
    st.markdown("### 📊 预测设置")
    forecast_days = st.number_input(
        "预测天数",
        min_value=1,
        max_value=30,
        value=7
    )

# 主界面
if uploaded_file is not None:
    # 加载数据
    if st.session_state.df is None:
        with st.spinner('正在加载数据...'):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success('✅ 数据加载成功！')
    
    df = st.session_state.df
    
    # 数据概览标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📊 数据概览", "🤖 模型训练", "🔮 预测分析", "📈 历史分析"])
    
    with tab1:
        st.markdown("## 数据概览")
        
        # 基础信息
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("总记录数", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("平均客流", f"{df['顾客数'].mean():.0f} 人")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("最高客流", f"{df['顾客数'].max():,} 人")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("最低客流", f"{df['顾客数'].min():,} 人")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 数据预览
        st.markdown("### 📋 数据预览")
        st.dataframe(df.head(10), use_container_width=True)
        
        # 客流趋势图
        st.markdown("### 📈 客流趋势")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(pd.to_datetime(df['日期']), df['顾客数'], color='#3498db', linewidth=1.5)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('客流量', fontsize=12)
        ax.set_title('历史客流趋势', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # 统计分析
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 星期分布")
            weekday_stats = df.groupby('星期')['顾客数'].mean().sort_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(range(len(weekday_stats)), weekday_stats.values, 
                          color=['#3498db' if i < 5 else '#e74c3c' for i in range(7)])
            ax.set_xticks(range(7))
            ax.set_xticklabels(['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
            ax.set_ylabel('平均客流量')
            ax.set_title('各星期平均客流')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            
        with col2:
            st.markdown("### 📊 假日分布")
            holiday_stats = df.groupby('假日')['顾客数'].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#3498db', '#e74c3c', '#f39c12']
            bars = ax.bar(range(len(holiday_stats)), holiday_stats.values, 
                          color=colors[:len(holiday_stats)])
            ax.set_xticks(range(len(holiday_stats)))
            ax.set_xticklabels(holiday_stats.index)
            ax.set_ylabel('平均客流量')
            ax.set_title('假日类型平均客流')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab2:
        st.markdown("## 模型训练")
        
        if not st.session_state.model_trained:
            st.info("🔍 点击下方按钮开始训练模型")
            
            if st.button("🚀 开始训练", key="train_button"):
                # 显示友好的等待提示
                st.markdown("""
                <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                    <h4 style='color: #155724; margin: 0;'>🎯 模型训练已开始！</h4>
                    <p style='color: #155724; margin: 10px 0 0 0;'>
                        请耐心等待，训练过程大约需要 <strong>2-5分钟</strong>。<br>
                        训练期间您可以看到实时的训练进度和损失曲线。<br>
                        <em>提示：训练时间取决于数据量大小和参数设置。</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner('正在准备数据...'):
                    # 创建特征
                    df_features = st.session_state.predictor.create_features(df)
                    
                    # 准备数据
                    exclude_cols = ['日期', '门店名称', '天气', '星期', '假日', '顾客数', 
                                   '年', '月', '日', '季度']
                    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
                    st.session_state.predictor.feature_cols = feature_cols
                    
                    features = df_features[feature_cols].values
                    target = df_features['顾客数'].values
                    
                    # 标准化
                    features_scaled = st.session_state.predictor.feature_scaler.fit_transform(features)
                    target_scaled = st.session_state.predictor.scaler.fit_transform(target.reshape(-1, 1))
                    
                    # 创建序列
                    X, y = [], []
                    for i in range(sequence_length, len(features)):
                        X.append(features_scaled[i-sequence_length:i])
                        y.append(target_scaled[i])
                    
                    X = np.array(X)
                    y = np.array(y)
                    
                    # 划分数据
                    train_size = int(len(X) * 0.8)
                    X_train, X_val = X[:train_size], X[train_size:]
                    y_train, y_val = y[:train_size], y[train_size:]
                
                # 训练进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 构建模型
                with st.spinner('正在构建模型架构...'):
                    model = st.session_state.predictor.build_model((X.shape[1], X.shape[2]))
                    st.session_state.predictor.model = model
                
                # 训练历史记录
                st.markdown("### 📊 训练监控")
                col1, col2 = st.columns(2)
                loss_placeholder = col1.empty()
                mae_placeholder = col2.empty()
                
                # 自定义回调
                class StreamlitCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f'训练进度: {epoch+1}/{epochs} 轮')
                        
                        # 更新图表
                        if hasattr(self, 'history'):
                            self.history['loss'].append(logs['loss'])
                            self.history['val_loss'].append(logs['val_loss'])
                            self.history['mae'].append(logs['mae'])
                            self.history['val_mae'].append(logs['val_mae'])
                        else:
                            self.history = {
                                'loss': [logs['loss']],
                                'val_loss': [logs['val_loss']],
                                'mae': [logs['mae']],
                                'val_mae': [logs['val_mae']]
                            }
                        
                        # 绘制损失图
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        ax1.plot(self.history['loss'], label='训练损失', color='#3498db')
                        ax1.plot(self.history['val_loss'], label='验证损失', color='#e74c3c')
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Loss')
                        ax1.set_title('训练损失')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        plt.tight_layout()
                        loss_placeholder.pyplot(fig1)
                        plt.close()
                        
                        # 绘制MAE图
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        ax2.plot(self.history['mae'], label='训练MAE', color='#3498db')
                        ax2.plot(self.history['val_mae'], label='验证MAE', color='#e74c3c')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('MAE')
                        ax2.set_title('平均绝对误差')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        mae_placeholder.pyplot(fig2)
                        plt.close()
                
                # 训练模型
                callbacks = [
                    StreamlitCallback(),
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
                ]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # 评估模型
                y_pred_scaled = model.predict(X_val)
                y_val_original = st.session_state.predictor.scaler.inverse_transform(y_val)
                y_pred_original = st.session_state.predictor.scaler.inverse_transform(y_pred_scaled)
                
                mae = mean_absolute_error(y_val_original, y_pred_original)
                rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
                r2 = r2_score(y_val_original, y_pred_original)
                
                # 计算MAPE
                mask = y_val_original.flatten() != 0
                mape = np.mean(np.abs((y_val_original[mask] - y_pred_original[mask]) / y_val_original[mask])) * 100
                
                # 显示结果
                st.success('✅ 模型训练完成！')
                
                # 训练结果展示
                st.markdown("### 🎯 训练结果")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{mae:.2f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col3:
                    st.metric("R²", f"{r2:.4f}")
                with col4:
                    st.metric("MAPE", f"{mape:.2f}%")
                
                # 成功提示
                st.markdown("""
                <div style='background-color: #d1ecf1; padding: 15px; border-radius: 10px; margin: 20px 0;'>
                    <h4 style='color: #0c5460; margin: 0;'>🎉 训练成功！</h4>
                    <p style='color: #0c5460; margin: 10px 0 0 0;'>
                        模型已经准备就绪，您现在可以：<br>
                        • 前往 <strong>"预测分析"</strong> 标签页生成未来客流预测<br>
                        • 查看上方的训练曲线了解模型收敛情况<br>
                        • 如果对结果不满意，可以调整参数后重新训练
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.model_trained = True
                st.session_state.df_features = df_features
                
        else:
            st.success("✅ 模型已训练完成！可以进行预测了。")
            
            # 显示模型信息
            st.markdown("""
            <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <h4 style='color: #004085; margin: 0;'>📌 模型状态</h4>
                <p style='color: #004085; margin: 5px 0;'>
                    模型已成功训练并保存在当前会话中。<br>
                    您可以前往"预测分析"标签页进行客流预测。
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 重新训练模型"):
                st.session_state.model_trained = False
                st.rerun()
    
    with tab3:
        st.markdown("## 未来预测")
        
        if st.session_state.model_trained:
            if st.button("🔮 预测未来客流", key="predict_button"):
                with st.spinner('正在生成预测...'):
                    predictor = st.session_state.predictor
                    df_features = st.session_state.df_features
                    
                    predictions = []
                    prediction_dates = []
                    
                    # 准备最后的序列
                    last_sequence = df_features[predictor.feature_cols].iloc[-sequence_length:].values
                    last_sequence_scaled = predictor.feature_scaler.transform(last_sequence)
                    
                    # 逐天预测
                    for day in range(forecast_days):
                        # 预测
                        X_pred = last_sequence_scaled.reshape(1, sequence_length, -1)
                        pred_scaled = predictor.model.predict(X_pred, verbose=0)
                        pred = predictor.scaler.inverse_transform(pred_scaled)[0, 0]
                        predictions.append(pred)
                        
                        # 计算新日期
                        next_date = df_features['日期'].iloc[-1] + timedelta(days=day+1)
                        prediction_dates.append(next_date)
                        
                        # 更新序列（简化版）
                        new_features = last_sequence[-1].copy()
                        new_features_scaled = predictor.feature_scaler.transform([new_features])
                        last_sequence_scaled = np.vstack([last_sequence_scaled[1:], new_features_scaled])
                    
                    # 创建预测结果
                    predictions_df = pd.DataFrame({
                        '日期': prediction_dates,
                        '预测客流': [int(p) for p in predictions],
                        '星期': ['周' + '一二三四五六日'[d.weekday()] for d in prediction_dates],
                        '类型': ['周末' if d.weekday() >= 5 else '工作日' for d in prediction_dates]
                    })
                    
                    # 添加置信区间
                    historical_std = df_features['顾客数'].std()
                    predictions_df['预测下限'] = (predictions_df['预测客流'] - 1.96 * historical_std * 0.1).astype(int)
                    predictions_df['预测上限'] = (predictions_df['预测客流'] + 1.96 * historical_std * 0.1).astype(int)
                
                # 显示预测结果
                st.success('✅ 预测完成！')
                
                # 预测统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均预测客流", f"{predictions_df['预测客流'].mean():.0f} 人")
                with col2:
                    weekday_avg = predictions_df[predictions_df['类型']=='工作日']['预测客流'].mean()
                    st.metric("工作日平均", f"{weekday_avg:.0f} 人" if not np.isnan(weekday_avg) else "无")
                with col3:
                    weekend_avg = predictions_df[predictions_df['类型']=='周末']['预测客流'].mean()
                    st.metric("周末平均", f"{weekend_avg:.0f} 人" if not np.isnan(weekend_avg) else "无")
                
                st.markdown("---")
                
                # 预测表格
                st.markdown("### 📊 详细预测结果")
                st.dataframe(predictions_df, use_container_width=True)
                
                # 可视化预测
                st.markdown("### 📈 预测可视化")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 历史数据
                hist_days = min(60, len(df_features))
                hist_data = df_features.tail(hist_days)
                ax.plot(hist_data['日期'], hist_data['顾客数'], 
                       'o-', label='历史客流', markersize=4, color='#3498db')
                
                # 预测数据
                ax.plot(predictions_df['日期'], predictions_df['预测客流'], 
                       's-', label='预测客流', markersize=6, linewidth=2, color='#e74c3c')
                
                # 置信区间
                ax.fill_between(predictions_df['日期'], 
                              predictions_df['预测下限'], 
                              predictions_df['预测上限'],
                              alpha=0.3, color='#e74c3c', label='95%置信区间')
                
                ax.set_xlabel('日期', fontsize=12)
                ax.set_ylabel('客流量', fontsize=12)
                ax.set_title('客流预测结果', fontsize=14, pad=20)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # 下载预测结果
                st.markdown("### 💾 下载预测结果")
                csv = predictions_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载CSV文件",
                    data=csv,
                    file_name=f"客流预测_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        else:
            st.warning("⚠️ 请先在'模型训练'标签页训练模型")
    
    with tab4:
        st.markdown("## 历史分析")
        
        # 月度分析
        df_temp = df.copy()
        df_temp['日期'] = pd.to_datetime(df_temp['日期'])
        df_temp['年月'] = df_temp['日期'].dt.to_period('M')
        monthly_stats = df_temp.groupby('年月')['顾客数'].agg(['mean', 'sum', 'count'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 月度平均客流
        ax1.plot(monthly_stats.index.astype(str), monthly_stats['mean'], 
                marker='o', color='#3498db', linewidth=2, markersize=6)
        ax1.set_title('月度平均客流趋势', fontsize=14, pad=10)
        ax1.set_xlabel('月份')
        ax1.set_ylabel('平均客流量')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 月度总客流
        ax2.bar(monthly_stats.index.astype(str), monthly_stats['sum'], 
               color='#2ecc71', alpha=0.7)
        ax2.set_title('月度总客流量', fontsize=14, pad=10)
        ax2.set_xlabel('月份')
        ax2.set_ylabel('总客流量')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 热力图分析
     
        
        # 创建周-小时热力图数据（这里用星期-月份代替）
        df_temp['月'] = df_temp['日期'].dt.month
        df_temp['星期数'] = df_temp['日期'].dt.dayofweek
        heatmap_data = df_temp.pivot_table(
            values='顾客数', 
            index='星期数', 
            columns='月', 
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': '平均客流量'}, ax=ax)
        ax.set_yticklabels(['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
        ax.set_xlabel('月份')
        ax.set_ylabel('星期')


else:
    # 欢迎页面
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>客流预测系统</h2>
        <p style='font-size: 18px; color: #7f8c8d; margin: 20px 0;'>
            采用双向LSTM，<br>
            能够预测未来客流趋势。
        </p>
        <div style='background-color: #f0f2f6; padding: 30px; border-radius: 10px; margin: 30px auto; max-width: 600px;'>
            <h3>🚀 快速开始</h3>
            <ol style='text-align: left; font-size: 16px;'>
                <li>在左侧边栏上传您的客流数据文件（Excel或CSV格式）</li>
                <li>系统将自动加载并展示数据概览</li>
                <li>点击"模型训练"标签页开始训练预测模型</li>
                <li>训练完成后，在"预测分析"中查看未来客流预测</li>
            </ol>
        </div>
        <div style='margin-top: 30px;'>
            <h4>📊 数据格式要求</h4>
            <p>您的数据文件应包含以下列：</p>
            <ul style='text-align: left; display: inline-block;'>
                <li>日期</li>
                <li>顾客数</li>
                <li>星期</li>
                <li>假日（工作日/周末/节假日）</li>
                <li>门店名称（可选）</li>
                <li>天气（可选）</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>💡 Powered by TensorFlow & Streamlit | 🔧 双向LSTM + 注意力机制</p>
</div>
""", unsafe_allow_html=True)