import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

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
        # 计算注意力分数
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.u), axis=1)
        
        # 应用注意力权重
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class AdvancedLSTMPredictor:
    """高级双向LSTM+注意力机制预测模型"""
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.attention_weights = None
        
    def create_advanced_features(self, df):
        """创建高级特征"""
        print("创建高级特征...")
        
        # 确保日期格式正确
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        
        # 基础时间特征
        df['年'] = df['日期'].dt.year
        df['月'] = df['日期'].dt.month
        df['日'] = df['日期'].dt.day
        df['星期'] = df['日期'].dt.dayofweek
        df['季度'] = df['日期'].dt.quarter
        df['年中周数'] = df['日期'].dt.isocalendar().week
        
        # 高级周期性编码
        # 年度周期
        df['年度进度'] = df['日期'].dt.dayofyear / 365
        df['年度_sin'] = np.sin(2 * np.pi * df['年度进度'])
        df['年度_cos'] = np.cos(2 * np.pi * df['年度进度'])
        
        # 月度周期
        df['月_sin'] = np.sin(2 * np.pi * df['月'] / 12)
        df['月_cos'] = np.cos(2 * np.pi * df['月'] / 12)
        df['月_sin2'] = np.sin(4 * np.pi * df['月'] / 12)  # 二次谐波
        df['月_cos2'] = np.cos(4 * np.pi * df['月'] / 12)
        
        # 周周期
        df['星期_sin'] = np.sin(2 * np.pi * df['星期'] / 7)
        df['星期_cos'] = np.cos(2 * np.pi * df['星期'] / 7)
        
        # 日周期
        df['日_sin'] = np.sin(2 * np.pi * df['日'] / 30)
        df['日_cos'] = np.cos(2 * np.pi * df['日'] / 30)
        
        # 特殊日期特征
        df['是否周末'] = (df['星期'] >= 5).astype(int)
        df['是否周一'] = (df['星期'] == 0).astype(int)
        df['是否周五'] = (df['星期'] == 4).astype(int)
        df['是否月初'] = (df['日'] <= 5).astype(int)
        df['是否月中'] = ((df['日'] > 10) & (df['日'] <= 20)).astype(int)
        df['是否月末'] = (df['日'] >= 25).astype(int)
        
        # 假日特征
        holiday_map = {'工作日': 0, '周末': 1, '节假日': 2}
        df['假日类型'] = df['假日'].map(holiday_map)
        
        # 客流统计特征
        # 移动平均
        for window in [3, 7, 14, 21, 30]:
            df[f'MA{window}'] = df['顾客数'].rolling(window=window, min_periods=1).mean()
            
        # 指数加权移动平均
        for span in [7, 14, 21]:
            df[f'EMA{span}'] = df['顾客数'].ewm(span=span, adjust=False).mean()
            
        # 移动标准差（波动性）
        for window in [7, 14, 21]:
            df[f'STD{window}'] = df['顾客数'].rolling(window=window, min_periods=1).std().fillna(0)
            
        # 移动最大最小值
        df['MAX7'] = df['顾客数'].rolling(window=7, min_periods=1).max()
        df['MIN7'] = df['顾客数'].rolling(window=7, min_periods=1).min()
        df['RANGE7'] = df['MAX7'] - df['MIN7']
        
        # 变化率特征
        df['日环比'] = df['顾客数'].pct_change(1).fillna(0)
        df['周环比'] = df['顾客数'].pct_change(7).fillna(0)
        
        # 趋势特征
        df['趋势7'] = df['MA7'] - df['MA14']
        df['趋势14'] = df['MA14'] - df['MA30']
        
        # 相对位置特征
        df['相对MA7'] = df['顾客数'] / df['MA7'].replace(0, 1)
        df['相对MA30'] = df['顾客数'] / df['MA30'].replace(0, 1)
        
        # 删除前面的空值行
        df = df.dropna()
        
        print(f"特征工程完成，共 {len(df.columns)} 个特征")
        
        return df
    
    def build_bidirectional_lstm_attention(self, input_shape):
        """构建双向LSTM+注意力机制模型"""
        print("\n构建双向LSTM+注意力机制模型...")
        
        # 输入层
        inputs = Input(shape=input_shape)
        
        # 第一层双向LSTM
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
        )(inputs)
        x = layers.BatchNormalization()(x)
        
        # 第二层双向LSTM
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
        )(x)
        x = layers.BatchNormalization()(x)
        
        # 注意力层
        attention_output, attention_weights = AttentionLayer(32)(x)
        
        # 结合注意力输出和最后时间步输出
        lstm_output = layers.Bidirectional(
            layers.LSTM(32, return_sequences=False)
        )(x)
        
        # 连接注意力输出和LSTM输出
        combined = layers.concatenate([attention_output, lstm_output])
        
        # 全连接层
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # 输出层
        outputs = layers.Dense(1)(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 使用自定义学习率调度
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='huber',  # Huber损失对异常值更鲁棒
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_data(self, df):
        """准备训练数据"""
        print("\n准备数据...")
        
        # 选择特征（排除原始列）
        exclude_cols = ['日期', '门店名称', '天气', '星期', '假日', '顾客数', 
                       '年', '月', '日', '季度', '年中周数']


        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 准备数据
        features = df[feature_cols].values
        target = df['顾客数'].values
        
        # 标准化
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1))
        
        # 创建序列
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_scaled[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"数据形状 - X: {X.shape}, y: {y.shape}")
        
        return X, y, feature_cols
    
    def train_model(self, X, y, epochs=100, batch_size=32):
        """训练模型"""
        print("\n开始训练...")
        
        # 划分数据
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # 构建模型
        self.model = self.build_bidirectional_lstm_attention((X.shape[1], X.shape[2]))
        print(self.model.summary())
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 训练
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估验证集
        self.evaluate_model(X_val, y_val)
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """评估模型"""
        print("\n评估模型...")
        
        # 预测
        y_pred_scaled = self.model.predict(X_test)
        
        # 反标准化
        y_test_original = self.scaler.inverse_transform(y_test)
        y_pred_original = self.scaler.inverse_transform(y_pred_scaled)
        
        # 计算指标
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        r2 = r2_score(y_test_original, y_pred_original)
        
        # MAPE
        mask = y_test_original.flatten() != 0
        mape = np.mean(np.abs((y_test_original[mask] - y_pred_original[mask]) / y_test_original[mask])) * 100
        
        print(f"\n模型性能:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # 可视化预测结果
        self.plot_predictions(y_test_original, y_pred_original)
        
        return y_test_original, y_pred_original
    
    def plot_predictions(self, y_true, y_pred):
        """绘制预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 时间序列对比
        ax1 = axes[0, 0]
        n = min(100, len(y_true))
        ax1.plot(y_true[-n:], 'b-', label='实际值', linewidth=2)
        ax1.plot(y_pred[-n:], 'r--', label='预测值', linewidth=2)
        ax1.set_title('最后100个预测结果', fontsize=14)
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('客流量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 散点图
        ax2 = axes[0, 1]
        ax2.scatter(y_true, y_pred, alpha=0.5)
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax2.set_title('预测值 vs 实际值', fontsize=14)
        ax2.set_xlabel('实际客流量')
        ax2.set_ylabel('预测客流量')
        ax2.grid(True, alpha=0.3)
        
        # 3. 误差分布
        ax3 = axes[1, 0]
        errors = y_true.flatten() - y_pred.flatten()
        ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax3.set_title('预测误差分布', fontsize=14)
        ax3.set_xlabel('误差')
        ax3.set_ylabel('频次')
        ax3.grid(True, alpha=0.3)
        
        # 4. 相对误差
        ax4 = axes[1, 1]
        relative_errors = np.abs((y_true - y_pred) / y_true) * 100
        ax4.plot(relative_errors[-n:], 'g-', linewidth=1)
        ax4.axhline(y=10, color='r', linestyle='--', label='10% 误差线')
        ax4.set_title('相对误差百分比', fontsize=14)
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('误差百分比 (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future_advanced(self, df, feature_cols, days=7):
        """高级未来预测"""
        
        
        print(f"\n使用双向LSTM+注意力机制预测未来 {days} 天...")
        
        predictions = []
        prediction_dates = []
        
        # 准备最后的序列
        last_sequence = df[feature_cols].iloc[-self.sequence_length:].values
        last_sequence_scaled = self.feature_scaler.transform(last_sequence)
        
        # 逐天预测
        for day in range(days):
            # 预测
            X_pred = last_sequence_scaled.reshape(1, self.sequence_length, -1)
            pred_scaled = self.model.predict(X_pred, verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred)
            
            # 计算新日期
            next_date = df['日期'].iloc[-1] + timedelta(days=day+1)
            prediction_dates.append(next_date)
            
            # 创建新特征行
            new_features = self.create_future_features(df, predictions, next_date)
            

            new_features_array = [new_features[col] for col in feature_cols]
            new_features_scaled = self.feature_scaler.transform([new_features_array])
            
            
            
            # 更新序列
            last_sequence_scaled = np.vstack([last_sequence_scaled[1:], new_features_scaled])
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            '日期': prediction_dates,
            '预测客流': [int(p) for p in predictions],
            '星期': ['周' + '一二三四五六日'[d.weekday()] for d in prediction_dates],
            '类型': ['周末' if d.weekday() >= 5 else '工作日' for d in prediction_dates]
        })
        
        # 添加置信区间（基于历史误差）
        historical_std = df['顾客数'].std()
        result['预测下限'] = (result['预测客流'] - 1.96 * historical_std * 0.1).astype(int)
        result['预测上限'] = (result['预测客流'] + 1.96 * historical_std * 0.1).astype(int)
        
        return result
    
    def create_future_features(self, df, recent_predictions, next_date):
        """为未来日期创建特征"""
        # 获取最近的客流数据（包括预测值）
        recent_values = list(df['顾客数'].iloc[-30:]) + recent_predictions

        features = {
            # 添加年份和月份特征
            '年份': next_date.year,  # 新增年份特征
            '月份': next_date.month,  # 新增月份特征

            # 年度周期
            '年度进度': next_date.timetuple().tm_yday / 365,
            '年度_sin': np.sin(2 * np.pi * next_date.timetuple().tm_yday / 365),
            '年度_cos': np.cos(2 * np.pi * next_date.timetuple().tm_yday / 365),

            # 月度周期
            '月_sin': np.sin(2 * np.pi * next_date.month / 12),
            '月_cos': np.cos(2 * np.pi * next_date.month / 12),
            '月_sin2': np.sin(4 * np.pi * next_date.month / 12),
            '月_cos2': np.cos(4 * np.pi * next_date.month / 12),
        
            # 周周期
            '星期_sin': np.sin(2 * np.pi * next_date.dayofweek / 7),
            '星期_cos': np.cos(2 * np.pi * next_date.dayofweek / 7),

            # 日周期
            '日_sin': np.sin(2 * np.pi * next_date.day / 30),
            '日_cos': np.cos(2 * np.pi * next_date.day / 30),

            # 特殊日期
            '是否周末': 1 if next_date.dayofweek >= 5 else 0,
            '是否周一': 1 if next_date.dayofweek == 0 else 0,
            '是否周五': 1 if next_date.dayofweek == 4 else 0,
            '是否月初': 1 if next_date.day <= 5 else 0,
            '是否月中': 1 if 10 < next_date.day <= 20 else 0,
            '是否月末': 1 if next_date.day >= 25 else 0,

            # 假日类型（简化）
            '假日类型': 1 if next_date.dayofweek >= 5 else 0,
        }

        # 移动平均
        for window in [3, 7, 14, 21, 30]:
            if len(recent_values) >= window:
                features[f'MA{window}'] = np.mean(recent_values[-window:])
            else:
                features[f'MA{window}'] = np.mean(recent_values)

        # 指数加权移动平均（简化计算）
        features['EMA7'] = features['MA7']
        features['EMA14'] = features['MA14']
        features['EMA21'] = features['MA21']

        # 标准差
        for window in [7, 14, 21]:
            if len(recent_values) >= window:
                features[f'STD{window}'] = np.std(recent_values[-window:])
            else:
                features[f'STD{window}'] = np.std(recent_values)

        # 最大最小值
        if len(recent_values) >= 7:
            features['MAX7'] = max(recent_values[-7:])
            features['MIN7'] = min(recent_values[-7:])
            features['RANGE7'] = features['MAX7'] - features['MIN7']
        else:
            features['MAX7'] = max(recent_values)
            features['MIN7'] = min(recent_values)
            features['RANGE7'] = features['MAX7'] - features['MIN7']

        # 变化率
        if len(recent_values) >= 2:
            features['日环比'] = (recent_values[-1] - recent_values[-2]) / recent_values[-2] if recent_values[-2] != 0 else 0
        else:
            features['日环比'] = 0

        if len(recent_values) >= 8:
            features['周环比'] = (recent_values[-1] - recent_values[-8]) / recent_values[-8] if recent_values[-8] != 0 else 0
        else:
            features['周环比'] = 0

        # 趋势
        features['趋势7'] = features['MA7'] - features['MA14']
        features['趋势14'] = features['MA14'] - features['MA30']

        # 相对位置
        features['相对MA7'] = recent_values[-1] / features['MA7'] if features['MA7'] != 0 else 1
        features['相对MA30'] = recent_values[-1] / features['MA30'] if features['MA30'] != 0 else 1

        return features
    

def main():
    """主函数"""
    # 初始化预测器
    predictor = AdvancedLSTMPredictor(sequence_length=30)
    
    # 加载数据
    print("="*60)
    print("高级双向LSTM+注意力机制客流预测")
    print("="*60)
    
    df = pd.read_excel('三福.xlsx')
    print(f"原始数据形状: {df.shape}")
    
    # 创建高级特征
    df = predictor.create_advanced_features(df)
    
    # 准备数据
    X, y, feature_cols = predictor.prepare_data(df)
    
    
    # 训练模型
    history = predictor.train_model(X, y, epochs=100, batch_size=32)
    


    # 预测未来
    future_predictions = predictor.predict_future_advanced(df, feature_cols, days=7)
    
    print("\n" + "="*60)
    print("未来7天预测结果（含置信区间）:")
    print("="*60)
    for _, row in future_predictions.iterrows():
        print(f"{row['日期'].strftime('%Y-%m-%d')} ({row['星期']}, {row['类型']})")
        print(f"  预测客流: {row['预测客流']} 人")
        print(f"  置信区间: [{row['预测下限']} - {row['预测上限']}]")
        print("-"*40)
    
    # 可视化未来预测
    plt.figure(figsize=(14, 7))
    
    # 历史数据
    hist_days = 60
    hist_data = df.tail(hist_days)
    plt.plot(hist_data['日期'], hist_data['顾客数'], 'o-', label='历史客流', markersize=4)
    
    # 预测数据
    plt.plot(future_predictions['日期'], future_predictions['预测客流'], 
             's-', label='预测客流', color='red', markersize=6, linewidth=2)
    
    # 置信区间
    plt.fill_between(future_predictions['日期'], 
                    future_predictions['预测下限'], 
                    future_predictions['预测上限'],
                    alpha=0.3, color='red', label='95%置信区间')
    
    plt.title('双向LSTM+注意力机制客流预测', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('客流量', fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    future_predictions.to_csv('advanced_lstm_predictions.csv', index=False, encoding='utf-8-sig')
    print("\n预测结果已保存到 'advanced_lstm_predictions.csv'")
    
    # 保存模型
    predictor.model.save('advanced_lstm_model.h5')
    print("模型已保存到 'advanced_lstm_model.h5'")

if __name__ == "__main__":
    main()