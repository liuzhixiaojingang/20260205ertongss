import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import io
import base64
import os
import shap
import networkx as nx
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import plotly.graph_objects as go
import plotly.io as pio

# 设置页面配置
st.set_page_config(page_title="儿童皮肤损伤识别系统", page_icon="👶🔥", layout="wide", initial_sidebar_state="expanded")

# 自定义CSS样式
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #ff6b35; text-align: center; margin-bottom: 2rem; font-weight: bold; font-family: "Microsoft YaHei", sans-serif; }
    .sub-header { font-size: 1.5rem; color: #ff8e53; margin: 1rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .feature-box { background-color: #fff5f5; padding: 1rem; border-radius: 10px; border-left: 4px solid #ff6b35; margin: 0.5rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .prediction-box { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .analysis-box { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #2196F3; margin: 1rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .setting-box { background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #627d98; margin: 0.5rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .guide-section { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #6c757d; font-family: "Microsoft YaHei", sans-serif; }
    .theory-box { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #ffc107; font-family: "Microsoft YaHei", sans-serif; }
    .code-box { background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 4px solid #6c757d; font-family: "Courier New", monospace; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# 加载模型函数（修改为本地模型）
@st.cache_resource
def load_model():
    try:
            import gdown
            # 从Google Drive下载模型
            model_url = "https://github.com/liuzhixiaojingang/20260205ertongss/raw/main/rf.pkl"
            model_path = "rf_model.pkl"
            
            if not os.path.exists(model_path):
                with st.spinner("正在从云端下载模型..."):
                    gdown.download(model_url, model_path, quiet=False)
            
            model = joblib.load(model_path)
            # 设置特征名称
            model.feature_names_in_ = ['BG1', 'Ascorbic acid', 'Pregnenolone sulfate', 'IL-1β', '5-Methoxytryptamine', 'EGF', 'BG2']
            st.success("✅ 本地模型加载成功")
            return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

# 获取图表字体设置函数
def get_chart_font_settings():
    """获取图表字体设置"""
    return {
        'title_font': st.session_state.get('chart_title_font', {'family': 'Microsoft YaHei', 'size': 14, 'weight': 'bold'}),
        'axis_font': st.session_state.get('chart_axis_font', {'family': 'Microsoft YaHei', 'size': 10}),
        'tick_font': st.session_state.get('chart_tick_font', {'family': 'Microsoft YaHei', 'size': 8}),
        'label_font': st.session_state.get('chart_label_font', {'family': 'Microsoft YaHei', 'size': 9})
    }

# 应用图表字体设置函数
def apply_chart_font_settings(ax=None, title=None, xlabel=None, ylabel=None):
    """应用图表字体设置"""
    font_settings = get_chart_font_settings()
    if ax is not None:
        if title and ax.get_title():
            ax.set_title(ax.get_title(), fontfamily=font_settings['title_font']['family'], fontsize=font_settings['title_font']['size'], fontweight=font_settings['title_font']['weight'])
        if xlabel or ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel() if not xlabel else xlabel, fontfamily=font_settings['axis_font']['family'], fontsize=font_settings['axis_font']['size'])
        if ylabel or ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel() if not ylabel else ylabel, fontfamily=font_settings['axis_font']['family'], fontsize=font_settings['axis_font']['size'])
        ax.tick_params(axis='both', which='major', labelsize=font_settings['tick_font']['size'])
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontfamily(font_settings['label_font']['family'])
                text.set_fontsize(font_settings['label_font']['size'])

# SHAP分析函数
def perform_shap_analysis(model, input_data, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        prediction = model.predict(input_data)[0]
        if shap_values.ndim == 3:
            current_shap_values = shap_values[0, :, prediction]
        else:
            st.error(f"不支持的SHAP维度: {shap_values.ndim}")
            return None
        if current_shap_values.ndim > 1: current_shap_values = current_shap_values[0]
        feature_importance = np.abs(current_shap_values)
        sorted_idx = np.argsort(feature_importance)[::-1]
        return {
            'shap_values': current_shap_values, 'shap_values_3d': shap_values, 'input_data': input_data,
            'feature_importance': feature_importance, 'sorted_features': [feature_names[i] for i in sorted_idx],
            'sorted_importance': feature_importance[sorted_idx], 'prediction': prediction
        }
    except Exception as e:
        st.error(f"SHAP分析错误: {str(e)}")
        return None

# 图1: 合并的SHAP分析图表
def plot_combined_shap_analysis(shap_results, feature_names, burn_type_mapping):
    try:
        if shap_results is None: return None
        shap_values_3d = shap_results['shap_values_3d']
        prediction = shap_results['prediction']
        font_settings = get_chart_font_settings()
        plt.rcParams.update({
            'font.size': font_settings['tick_font']['size'],
            'axes.titlesize': font_settings['title_font']['size'],
            'axes.labelsize': font_settings['axis_font']['size'],
            'xtick.labelsize': font_settings['tick_font']['size'],
            'ytick.labelsize': font_settings['tick_font']['size'],
            'font.family': font_settings['title_font']['family']
        })
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('SHAP Analysis: Feature Impact and Importance for All Classes', fontsize=font_settings['title_font']['size'] + 2, fontweight='bold', y=0.95, fontfamily=font_settings['title_font']['family'])
        for i in range(6):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            if shap_values_3d.ndim == 3:
                class_shap_values = np.mean(shap_values_3d[:, :, i], axis=0)
                class_shap_importance = np.mean(np.abs(shap_values_3d[:, :, i]), axis=0)
            else:
                class_shap_values = shap_values_3d[i]
                class_shap_importance = np.abs(shap_values_3d[i])
            sorted_idx = np.argsort(class_shap_importance)[::-1]
            sorted_features = [feature_names[j] for j in sorted_idx]
            sorted_shap = class_shap_values[sorted_idx]
            sorted_importance = class_shap_importance[sorted_idx]
            y_pos = np.arange(len(sorted_features))
            colors = ['#ff6b6b' if val > 0 else '#4ecdc4' for val in sorted_shap]
            bars = ax.barh(y_pos, sorted_shap, color=colors, alpha=0.8, height=0.6)
            for j, (shap_val, imp_val) in enumerate(zip(sorted_shap, sorted_importance)):
                ax.scatter(imp_val if shap_val >= 0 else -imp_val, j, s=80, color='#2d3436', marker='o', alpha=0.7, zorder=5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features, fontfamily=font_settings['tick_font']['family'])
            ax.invert_yaxis()
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
            ax.set_xlabel('SHAP Value / Importance', fontsize=font_settings['axis_font']['size'], fontweight='bold', fontfamily=font_settings['axis_font']['family'])
            ax.grid(True, alpha=0.3, axis='x')
            if i == prediction:
                ax.patch.set_facecolor('#fffacd')
                ax.patch.set_alpha(0.3)
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
                title_color = 'red'
                title_suffix = ' ★'
            else:
                title_color = 'black'
                title_suffix = ''
            ax.set_title(f'Class {i}: {burn_type_mapping[i]["en"]}{title_suffix}', fontsize=font_settings['title_font']['size'], fontweight='bold', color=title_color, pad=10, fontfamily=font_settings['title_font']['family'])
            for j, (bar, shap_val, imp_val) in enumerate(zip(bars, sorted_shap, sorted_importance)):
                width = bar.get_width()
                if abs(shap_val) > 0.001:
                    if shap_val > 0:
                        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2., f'{shap_val:+.6f}', ha='left', va='center', fontsize=font_settings['label_font']['size'] - 1, color='#d63031', fontweight='bold', fontfamily=font_settings['label_font']['family'])
                    else:
                        ax.text(width - 0.005, bar.get_y() + bar.get_height()/2., f'{shap_val:+.6f}', ha='right', va='center', fontsize=font_settings['label_font']['size'] - 1, color='#00b894', fontweight='bold', fontfamily=font_settings['label_font']['family'])
                    ax.text(imp_val + 0.005 if shap_val >= 0 else -imp_val - 0.005, j, f'{imp_val:.6f}', ha='left' if shap_val >= 0 else 'right', va='center', fontsize=font_settings['label_font']['size'] - 2, color='#2d3436', fontweight='bold', fontfamily=font_settings['label_font']['family'])
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff6b6b', alpha=0.8, label='Positive Impact'),
            Patch(facecolor='#4ecdc4', alpha=0.8, label='Negative Impact'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2d3436', markersize=6, label='Importance Magnitude')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=font_settings['label_font']['size'], framealpha=0.9, fancybox=True, shadow=True)
        return fig
    except Exception as e:
        st.error(f"SHAP图表绘制错误: {str(e)}")
        return None

# 图2: 当前预测类别的特征重要性图
def plot_current_prediction_shap(shap_results, feature_names, burn_type_mapping):
    try:
        if shap_results is None: return None
        prediction = shap_results['prediction']
        sorted_features = shap_results['sorted_features']
        sorted_importance = shap_results['sorted_importance']
        font_settings = get_chart_font_settings()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'SHAP Analysis for Current Prediction: {burn_type_mapping[prediction]["en"]}', fontsize=font_settings['title_font']['size'] + 2, fontweight='bold', fontfamily=font_settings['title_font']['family'])
        y_pos = np.arange(len(sorted_features))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
        bars = ax1.barh(y_pos, sorted_importance, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_features, fontfamily=font_settings['tick_font']['family'])
        ax1.invert_yaxis()
        ax1.set_xlabel('SHAP Value Importance', fontweight='bold', fontfamily=font_settings['axis_font']['family'], fontsize=font_settings['axis_font']['size'])
        ax1.set_title('Feature Importance Ranking', fontweight='bold', fontfamily=font_settings['title_font']['family'], fontsize=font_settings['title_font']['size'])
        ax1.grid(True, alpha=0.3, axis='x')
        for bar, importance in zip(bars, sorted_importance):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2., f'{width:.10f}', ha='left', va='center', fontsize=font_settings['label_font']['size'], fontweight='bold', fontfamily=font_settings['label_font']['family'])
        shap_values = shap_results['shap_values']
        positive_count = np.sum(shap_values > 0)
        negative_count = np.sum(shap_values < 0)
        neutral_count = np.sum(shap_values == 0)
        sizes = [positive_count, negative_count, neutral_count]
        labels = ['Positive Impact', 'Negative Impact', 'No Impact']
        colors = ['#ff6b6b', '#4ecdc4', '#95a5a6']
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontfamily': font_settings['label_font']['family'], 'fontsize': font_settings['label_font']['size']})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax2.text(0.5, 0.5, 'No significant\nSHAP values', ha='center', va='center', transform=ax2.transAxes, fontsize=font_settings['label_font']['size'], fontfamily=font_settings['label_font']['family'])
        ax2.set_title('SHAP Value Distribution', fontweight='bold', fontfamily=font_settings['title_font']['family'], fontsize=font_settings['title_font']['size'])
        apply_chart_font_settings(ax1, xlabel='SHAP Value Importance')
        apply_chart_font_settings(ax2)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"当前预测SHAP图表绘制错误: {str(e)}")
        return None

# 优化的图网络分析
def perform_graph_analysis(feature_values, feature_names, prediction, burn_type_mapping):
    try:
        G = nx.Graph()
        for i, feature in enumerate(feature_names):
            G.add_node(feature, value=feature_values[i], importance=abs(feature_values[i]))
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                correlation = 1 - abs(feature_values[i] - feature_values[j]) / (abs(feature_values[i]) + abs(feature_values[j]) + 1e-8)
                if correlation > 0.3:
                    G.add_edge(feature_names[i], feature_names[j], weight=correlation)
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        return {
            'graph': G, 'degree_centrality': degree_centrality, 'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality, 'node_importance': {feature: abs(val) for feature, val in zip(feature_names, feature_values)}
        }
    except Exception as e:
        st.warning(f"图网络分析遇到问题: {str(e)}")
        return None

# 优化的图网络可视化
def plot_optimized_graph_analysis(graph_results, feature_names, burn_info):
    try:
        if graph_results is None: return None
        G = graph_results['graph']
        font_settings = get_chart_font_settings()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Feature Network Analysis - {burn_info["cn"]}', fontsize=font_settings['title_font']['size'] + 2, fontweight='bold', fontfamily=font_settings['title_font']['family'])
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')
        pos = nx.spring_layout(G, seed=42, k=3, iterations=200)
        node_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD700', '#9370DB', '#20B2AA']
        node_color_map = {feature: node_colors[i] for i, feature in enumerate(feature_names)}
        node_sizes = [3000 + 2000 * graph_results['node_importance'][node] for node in G.nodes()]
        node_colors_list = [node_color_map[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors_list, alpha=0.9, ax=ax1, edgecolors='black', linewidths=2)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]
        edge_colors = ['#2C3E50' for _ in edges]
        edge_widths = [w * 5 + 1 for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=[min(w * 1.5, 0.8) for w in weights], edge_color=edge_colors, ax=ax1, style='solid')
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold', ax=ax1, font_family=font_settings['label_font']['family'], bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        ax1.set_title('Network Topology', fontsize=font_settings['title_font']['size'], fontweight='bold', fontfamily=font_settings['title_font']['family'])
        ax1.axis('off')
        centrality_data = {
            'Feature': list(graph_results['degree_centrality'].keys()),
            'Degree': list(graph_results['degree_centrality'].values()),
            'Betweenness': list(graph_results['betweenness_centrality'].values()),
            'Closeness': list(graph_results['closeness_centrality'].values())
        }
        df = pd.DataFrame(centrality_data)
        categories = list(df['Feature'])
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        ax2 = plt.subplot(132, polar=True)
        ax2.set_facecolor('white')
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, fontfamily=font_settings['tick_font']['family'])
        values = df['Degree'].values.tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label='Degree Centrality', color='#e74c3c')
        ax2.fill(angles, values, alpha=0.25, color='#e74c3c')
        values = df['Betweenness'].values.tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label='Betweenness Centrality', color='#3498db')
        ax2.fill(angles, values, alpha=0.25, color='#3498db')
        values = df['Closeness'].values.tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label='Closeness Centrality', color='#2ecc71')
        ax2.fill(angles, values, alpha=0.25, color='#2ecc71')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop={'family': font_settings['label_font']['family'], 'size': font_settings['label_font']['size']})
        ax2.set_title('Centrality Analysis Radar Chart', fontsize=font_settings['title_font']['size'], fontweight='bold', fontfamily=font_settings['title_font']['family'])
        ax3.set_facecolor('white')
        correlation_matrix = np.zeros((len(feature_names), len(feature_names)))
        for i, feat1 in enumerate(feature_names):
            for j, feat2 in enumerate(feature_names):
                if feat1 == feat2:
                    correlation_matrix[i, j] = 1.0
                elif G.has_edge(feat1, feat2):
                    correlation_matrix[i, j] = G[feat1][feat2]['weight']
                else:
                    correlation_matrix[i, j] = 0.0
        im = ax3.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax3.set_xticks(range(len(feature_names)))
        ax3.set_yticks(range(len(feature_names)))
        ax3.set_xticklabels(feature_names, rotation=45, fontfamily=font_settings['tick_font']['family'])
        ax3.set_yticklabels(feature_names, fontfamily=font_settings['tick_font']['family'])
        ax3.set_title('Feature Correlation Heatmap', fontsize=font_settings['title_font']['size'], fontweight='bold', fontfamily=font_settings['title_font']['family'])
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.6f}', ha="center", va="center", color="black", fontsize=font_settings['label_font']['size'] - 1, fontweight='bold', fontfamily=font_settings['label_font']['family'])
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        apply_chart_font_settings(ax1)
        apply_chart_font_settings(ax2)
        apply_chart_font_settings(ax3)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"图网络可视化错误: {str(e)}")
        return None

# 反事实分析函数
def perform_counterfactual_analysis(model, input_data, original_prediction, feature_names, burn_type_mapping):
    try:
        if original_prediction == 0:
            return {
                'all_counterfactuals': [],
                'normal_tissue_suggestions': [],
                'original_prediction': original_prediction,
                'skip_analysis': True
            }
        base_values = input_data.iloc[0].values
        counterfactuals = []
        normal_tissue_suggestions = []
        for i, feature in enumerate(feature_names):
            for change_factor in [0.5, 0.7, 1.3, 1.5, 2.0]:
                modified_data = base_values.copy()
                modified_data[i] = modified_data[i] * change_factor
                modified_df = pd.DataFrame([modified_data], columns=feature_names)
                new_prediction = model.predict(modified_df)[0]
                new_probability = model.predict_proba(modified_df)[0][new_prediction]
                if new_prediction != original_prediction:
                    counterfactuals.append({
                        'changed_feature': feature, 'change_factor': change_factor,
                        'new_prediction': new_prediction, 'confidence': new_probability,
                        'required_change': f"{change_factor:.1f}x", 'original_value': base_values[i],
                        'new_value': modified_data[i], 'change_direction': "增加" if change_factor > 1 else "减少"
                    })
                if new_prediction == 0:
                    normal_tissue_suggestions.append({
                        'feature': feature, 'change_factor': change_factor,
                        'confidence': new_probability, 'required_change': f"{change_factor:.1f}x",
                        'original_value': base_values[i], 'new_value': modified_data[i],
                        'change_direction': "增加" if change_factor > 1 else "减少"
                    })
        normal_tissue_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        counterfactuals.sort(key=lambda x: x['confidence'], reverse=True)
        return {
            'all_counterfactuals': counterfactuals[:5],
            'normal_tissue_suggestions': normal_tissue_suggestions[:3],
            'original_prediction': original_prediction,
            'skip_analysis': False
        }
    except Exception as e:
        st.warning(f"反事实分析遇到问题: {str(e)}")
        return {'all_counterfactuals': [], 'normal_tissue_suggestions': [], 'original_prediction': original_prediction, 'skip_analysis': False}

# 优化的反事实分析可视化
def plot_optimized_counterfactual_analysis(counterfactual_results, burn_type_mapping):
    try:
        if not counterfactual_results or counterfactual_results.get('skip_analysis', False):
            return None
        suggestions = counterfactual_results['normal_tissue_suggestions']
        if not suggestions:
            return None
        font_settings = get_chart_font_settings()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Counterfactual Analysis - Normal Tissue Recovery Strategies', fontsize=font_settings['title_font']['size'] + 2, fontweight='bold', fontfamily=font_settings['title_font']['family'])
        features = [s['feature'] for s in suggestions]
        confidences = [s['confidence'] for s in suggestions]
        change_factors = [s['change_factor'] for s in suggestions]
        colors = ['#4CAF50' if factor > 1 else '#F44336' for factor in change_factors]
        bars = ax1.barh(features, confidences, color=colors, alpha=0.8, height=0.6)
        ax1.set_xlabel('Confidence Level', fontweight='bold', fontfamily=font_settings['axis_font']['family'], fontsize=font_settings['axis_font']['size'])
        ax1.set_title('Recovery Strategy Effectiveness', fontsize=font_settings['title_font']['size'], fontweight='bold', fontfamily=font_settings['title_font']['family'])
        ax1.set_xlim(0, 1)
        ax1.grid(True, alpha=0.3, axis='x')
        for bar, factor, conf in zip(bars, change_factors, confidences):
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{factor:.1f}x\n{conf:.1%}', ha='left', va='center', fontweight='bold', fontsize=font_settings['label_font']['size'], fontfamily=font_settings['label_font']['family'])
        features = [s['feature'] for s in suggestions]
        original_vals = [s['original_value'] for s in suggestions]
        target_vals = [s['new_value'] for s in suggestions]
        changes = [s['change_factor'] for s in suggestions]
        x_pos = np.arange(len(features))
        width = 0.35
        bars1 = ax2.bar(x_pos - width/2, original_vals, width, label='Current Value', color='#2196F3', alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, target_vals, width, label='Target Value', color='#4CAF50', alpha=0.7)
        for i, (orig, target, change) in enumerate(zip(original_vals, target_vals, changes)):
            arrow_x = i
            arrow_y1 = orig
            arrow_y2 = target
            arrow_color = 'red' if change > 1 else 'blue'
            arrow_style = '->' if change > 1 else '<-'
            ax2.annotate('', xy=(arrow_x + width/2, arrow_y2), xytext=(arrow_x - width/2, arrow_y1), arrowprops=dict(arrowstyle=arrow_style, color=arrow_color, lw=2))
        ax2.set_xlabel('Features', fontweight='bold', fontfamily=font_settings['axis_font']['family'], fontsize=font_settings['axis_font']['size'])
        ax2.set_ylabel('Values', fontweight='bold', fontfamily=font_settings['axis_font']['family'], fontsize=font_settings['axis_font']['size'])
        ax2.set_title('Feature Adjustment Pathways', fontsize=font_settings['title_font']['size'], fontweight='bold', fontfamily=font_settings['title_font']['family'])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(features, rotation=45, fontfamily=font_settings['tick_font']['family'])
        ax2.legend(prop={'family': font_settings['label_font']['family'], 'size': font_settings['label_font']['size']})
        ax2.grid(True, alpha=0.3)
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(target_vals), f'{height:.10f}', ha='center', va='bottom', fontsize=font_settings['label_font']['size'] - 1, fontweight='bold', fontfamily=font_settings['label_font']['family'])
        apply_chart_font_settings(ax1, xlabel='Confidence Level')
        apply_chart_font_settings(ax2, xlabel='Features', ylabel='Values')
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"反事实图表绘制错误: {str(e)}")
        return None

# 生成医疗检测报告的函数
def generate_medical_report(input_data, prediction, probabilities, shap_results, graph_results, counterfactual_results, burn_type_mapping, feature_names, language='中文'):
    """生成详细的医疗检测报告"""
    burn_info = burn_type_mapping[prediction]
    if language == '中文':
        report = f"""烧伤智能识别系统 - 医疗检测报告
==================================================
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
【基本信息】
患者样本编号: {pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}
分析模型: 随机森林多分类模型
数据精度: 小数点后10位
【输入参数详细数据】
BG1 (生物标志物1): {input_data.iloc[0, 0]:.10f}
Ascorbic acid (抗坏血酸): {input_data.iloc[0, 1]:.10f}
Pregnenolone sulfate (孕烯醇酮硫酸酯): {input_data.iloc[0, 2]:.10f}
IL-1β (白细胞介素-1β): {input_data.iloc[0, 3]:.10f} pg/mL
5-Methoxytryptamine (5-甲氧基色胺): {input_data.iloc[0, 4]:.10f}
EGF (表皮生长因子): {input_data.iloc[0, 5]:.10f} pg/mL
BG2 (生物标志物2): {input_data.iloc[0, 6]:.10f}
【诊断结果】
主要诊断: {burn_info['cn']} ({burn_info['en']})
置信度: {probabilities[prediction]:.2%}
临床描述: {burn_info['description']}
【概率分布分析】
"""
        for i, prob in enumerate(probabilities):
            report += f"{burn_type_mapping[i]['cn']}: {prob:.2%}\n"
        report += f"\n【生物标志物临床意义分析】\n" + "="*50 + "\n"
        if shap_results:
            shap_values = shap_results['shap_values']
            for i, feature in enumerate(feature_names):
                shap_val = shap_values[i]
                original_val = input_data.iloc[0, i]
                report += f"\n{feature}分析:\n"
                report += f"- 当前水平: {original_val:.10f}\n"
                report += f"- 对诊断影响: {shap_val:+.6f} "
                if shap_val > 0.01:
                    report += "(显著正向影响 → 促进该诊断)\n"
                elif shap_val < -0.01:
                    report += "(显著负向影响 → 抑制该诊断)\n"
                else:
                    report += "(影响较小)\n"
        if shap_results:
            report += f"\n【SHAP可解释性分析】\n" + "="*50 + "\n"
            report += "特征重要性排序 (基于SHAP绝对值):\n"
            for i, (feature, importance) in enumerate(zip(shap_results['sorted_features'], shap_results['sorted_importance'])):
                report += f"{i+1}. {feature}: {importance:.10f}\n"
        if graph_results:
            report += f"\n【图网络分析结果】\n" + "="*50 + "\n"
            report += f"网络节点数: {len(graph_results['graph'].nodes())}\n"
            report += f"网络边数: {len(graph_results['graph'].edges())}\n"
            report += "特征中心性分析:\n"
            for feature in graph_results['degree_centrality']:
                report += f"- {feature}: 度中心性={graph_results['degree_centrality'][feature]:.6f}, 介数中心性={graph_results['betweenness_centrality'][feature]:.6f}, 紧密中心性={graph_results['closeness_centrality'][feature]:.6f}\n"
        if counterfactual_results and not counterfactual_results.get('skip_analysis', False) and counterfactual_results['normal_tissue_suggestions']:
            report += f"\n【反事实分析与治疗建议】\n" + "="*50 + "\n"
            report += "基于模型预测的干预策略分析:\n\n"
            for i, suggestion in enumerate(counterfactual_results['normal_tissue_suggestions'][:3], 1):
                report += f"治疗方案 {i}:\n"
                report += f"- 调整目标: 将{suggestion['feature']}{suggestion['change_direction']}{suggestion['required_change']}\n"
                report += f"- 具体数值: {suggestion['original_value']:.10f} → {suggestion['new_value']:.10f}\n"
                report += f"- 预期效果置信度: {suggestion['confidence']:.2%}\n"
                report += f"- 临床意义: 预测从{burn_type_mapping[counterfactual_results['original_prediction']]['cn']}恢复到正常组织\n\n"
        report += f"\n【临床治疗建议与注意事项】\n" + "="*50 + "\n"
        if prediction == 0:
            report += "当前诊断为正常组织，无需特殊治疗。\n"
            report += "建议:\n"
            report += "- 定期监测生物标志物水平\n"
            report += "- 保持健康生活方式\n"
            report += "- 避免烧伤风险因素\n"
        else:
            report += f"针对{burn_info['cn']}的治疗建议:\n"
            if prediction in [1, 2]:
                report += "- 立即进行伤口清洁和消毒\n"
                report += "- 使用适当的敷料保护创面\n"
                report += "- 考虑使用生长因子促进愈合\n"
                report += "- 定期更换敷料，监测感染迹象\n"
                report += "- 如IL-1β水平高，考虑抗炎治疗\n"
            elif prediction == 3:
                report += "- 需要外科清创和植皮手术\n"
                report += "- 全身抗感染治疗\n"
                report += "- 营养支持，促进组织修复\n"
                report += "- 疼痛管理和炎症控制\n"
                report += "- 长期康复和功能训练\n"
            elif prediction == 4:
                report += "- 评估深部组织损伤程度\n"
                report += "- 监测心电图和肌酸激酶\n"
                report += "- 积极清创，预防感染\n"
                report += "- 注意可能的并发症\n"
                report += "- 多学科团队协作治疗\n"
            elif prediction == 5:
                report += "- 评估吸入性损伤风险\n"
                report += "- 全面清创和烧伤护理\n"
                report += "- 预防感染和败血症\n"
                report += "- 营养支持和代谢管理\n"
                report += "- 心理支持和康复治疗\n"
        report += f"\n【报告说明】\n" + "="*50 + "\n"
        report += "1. 本报告基于机器学习模型分析生成，仅供参考\n"
        report += "2. 临床诊断需结合临床表现和医师判断\n"
        report += "3. 治疗建议需在专业医师指导下实施\n"
        report += "4. 定期随访和监测对治疗效果至关重要\n"
    else:
        report = f"""Burn Intelligent Recognition System - Medical Analysis Report
==================================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
【Basic Information】
Sample ID: {pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}
Analysis Model: Random Forest Multi-class Model
Data Precision: 10 decimal places
【Input Parameters】
BG1 (Biomarker 1): {input_data.iloc[0, 0]:.10f}
Ascorbic acid: {input_data.iloc[0, 1]:.10f}
Pregnenolone sulfate: {input_data.iloc[0, 2]:.10f}
IL-1β (Interleukin-1β): {input_data.iloc[0, 3]:.10f} pg/mL
5-Methoxytryptamine: {input_data.iloc[0, 4]:.10f}
EGF (Epidermal Growth Factor): {input_data.iloc[0, 5]:.10f} pg/mL
BG2 (Biomarker 2): {input_data.iloc[0, 6]:.10f}
【Diagnosis Results】
Primary Diagnosis: {burn_info['en']} ({burn_info['cn']})
Confidence: {probabilities[prediction]:.2%}
Clinical Description: {burn_info['description_en']}
【Probability Distribution Analysis】
"""
        for i, prob in enumerate(probabilities):
            report += f"{burn_type_mapping[i]['en']}: {prob:.2%}\n"
    return report

# 3D皮肤模型函数
def create_skin_3d_model_with_burn_depth(prediction=None, probabilities=None, burn_color='#FF4500', burn_opacity=0.7):
    """创建带有烧伤深度标注的3D皮肤模型"""
    fig = go.Figure()
    
    # 定义皮肤各层
    epidermis_depth = 0.2
    dermis_depth = 2.0
    subcutaneous_depth = 5.0
    
    # 绘制皮肤各层
    epidermis_vertices_x = [0, 10, 10, 0, 0, 10, 10, 0]
    epidermis_vertices_y = [0, 0, 10, 10, 0, 0, 10, 10]
    epidermis_vertices_z = [0, 0, 0, 0, epidermis_depth, epidermis_depth, epidermis_depth, epidermis_depth]
    epidermis_i = [0, 0, 0, 0, 5, 7, 7, 5, 6, 1, 4, 2]
    epidermis_j = [3, 1, 2, 4, 1, 6, 3, 4, 0, 2, 5, 3]
    epidermis_k = [1, 2, 3, 7, 4, 3, 2, 0, 5, 6, 0, 6]
    
    fig.add_trace(go.Mesh3d(
        x=epidermis_vertices_x, y=epidermis_vertices_y, z=epidermis_vertices_z,
        i=epidermis_i, j=epidermis_j, k=epidermis_k,
        name='表皮层 (0-0.2mm)', color='#FFFACD', opacity=0.6, showlegend=False
    ))
    
    dermis_vertices_z = [epidermis_depth] * 4 + [dermis_depth] * 4
    fig.add_trace(go.Mesh3d(
        x=epidermis_vertices_x, y=epidermis_vertices_y, z=dermis_vertices_z,
        i=epidermis_i, j=epidermis_j, k=epidermis_k,
        name='真皮层 (0.2-2.0mm)', color='#FF6B6B', opacity=0.7, showlegend=False
    ))
    
    subcutaneous_vertices_z = [dermis_depth] * 4 + [subcutaneous_depth] * 4
    fig.add_trace(go.Mesh3d(
        x=epidermis_vertices_x, y=epidermis_vertices_y, z=subcutaneous_vertices_z,
        i=epidermis_i, j=epidermis_j, k=epidermis_k,
        name='皮下组织 (2.0-5.0mm)', color='#FFA07A', opacity=0.5, showlegend=False
    ))
    
    # 分界面
    x_interface = np.linspace(0, 10, 20)
    y_interface = np.linspace(0, 10, 20)
    X_interface, Y_interface = np.meshgrid(x_interface, y_interface)
    
    Z_skin_surface = np.zeros_like(X_interface)
    fig.add_trace(go.Surface(z=Z_skin_surface, x=X_interface, y=Y_interface, name='皮肤表面',
        colorscale=[[0, '#FAEBD7'], [1, '#F5DEB3']], opacity=0.9, showscale=False, showlegend=False))
    
    Z_epidermis_dermis_interface = np.ones_like(X_interface) * epidermis_depth
    fig.add_trace(go.Surface(z=Z_epidermis_dermis_interface, x=X_interface, y=Y_interface, name='表皮-真皮分界面',
        colorscale=[[0, '#F0E68C'], [1, '#DAA520']], opacity=0.8, showscale=False, showlegend=False))
    
    Z_dermis_subcutaneous_interface = np.ones_like(X_interface) * dermis_depth
    fig.add_trace(go.Surface(z=Z_dermis_subcutaneous_interface, x=X_interface, y=Y_interface, name='真皮-皮下组织分界面',
        colorscale=[[0, '#FF6347'], [1, '#B22222']], opacity=0.8, showscale=False, showlegend=False))
    
    # 根据预测结果添加烧伤区域
    burn_regions = []
    if prediction is not None and prediction > 0:  # 不是正常组织
        burn_depth_map = {
            1: 0.8,   # 浅表部分厚度烧伤
            2: 1.5,   # 深层部分厚度烧伤
            3: 3.5,   # 全层厚度烧伤
            4: 4.0,   # 电击烧伤
            5: 2.5    # 火焰烧伤
        }
        burn_depth = burn_depth_map.get(prediction, 1.0)
        
        # 创建半球形状烧伤区域
        burn_center_x, burn_center_y = 5, 5
        radius = 2.0
        phi = np.linspace(0, np.pi/2, 15)  # 从0到90度
        theta = np.linspace(0, 2*np.pi, 30)

        # 创建半球网格
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        x = burn_center_x + radius * np.sin(phi_grid) * np.cos(theta_grid)
        y = burn_center_y + radius * np.sin(phi_grid) * np.sin(theta_grid)
        z = burn_depth * np.cos(phi_grid)  # 半球形状

        # 添加半球烧伤区域
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, '#FF4500'], [1, '#FF6347']],
            opacity=0.7,
            showscale=False,
            name=f'烧伤区域'
        ))
        
        # 修改Surface的颜色和透明度参数
        colorscale=[[0, burn_color], [1, burn_color]],
        opacity=burn_opacity,
        
        # 添加烧伤深度标注
        burn_regions.append({
            'x': burn_center_x, 'y': burn_center_y, 'z': burn_depth/2,
            'text': f"{burn_type_mapping[prediction]['cn']}\n深度: {burn_depth}mm\n置信度: {probabilities[prediction]:.2%}"
        })
    
    annotations = []
    
    # 皮肤层名称标注
    layer_labels = [
        dict(x=0, y=9, z=0.1, text="<b>表皮层</b><br>(0-0.2mm)", showarrow=True, arrowhead=1, arrowwidth=2, font=dict(size=12, color="#8B6914")),
        dict(x=0, y=9, z=1.0, text="<b>真皮层</b><br>(0.2-2.0mm)", showarrow=True, arrowhead=1, arrowwidth=2, font=dict(size=12, color="#8B0000")),
        dict(x=0, y=9, z=3.5, text="<b>皮下组织</b><br>(2.0-5.0mm)", showarrow=True, arrowhead=1, arrowwidth=2, font=dict(size=12, color="#8B4500")),
    ]
    
    annotations.extend(layer_labels)
    
    # 烧伤区域标注
    for region in burn_regions:
        annotations.append(dict(
            x=region['x'], y=region['y'], z=region['z'],
            text=region['text'],
            showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor='red',
            font=dict(size=12, color="red"), bgcolor="rgba(255, 255, 255, 0.8)"
        ))
    
    # 配置图表布局
    title_text = "3D皮肤模型 - 烧伤深度分类系统"
    if prediction is not None:
        burn_info = burn_type_mapping[prediction]
        title_text = f"3D皮肤模型 - 烧伤诊断结果: {burn_info['cn']} ({burn_info['en']})"
    
    fig.update_layout(
        title={'text': title_text, 'x': 0.5, 'font': dict(size=20, color='darkblue')},
        scene=dict(
            xaxis=dict(title='皮肤表面 (mm)', range=[0, 12], showgrid=True, gridcolor='lightgray', backgroundcolor='rgba(240, 240, 240, 0.1)'),
            yaxis=dict(title='皮肤表面 (mm)', range=[0, 12], showgrid=True, gridcolor='lightgray'),
            zaxis=dict(title='深度 (mm)', range=[subcutaneous_depth, 0], showgrid=True, gridcolor='lightgray'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=0, z=1)),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.8),
            annotations=annotations
        ),
        showlegend=False, width=1000, height=700
    )
    
    return fig

# 自动加载模型
if 'model' not in st.session_state:
    with st.spinner("正在加载模型..."): 
        st.session_state.model = load_model()

# 烧伤类型映射
burn_type_mapping = {
    0: {"en": "Normal", "cn": "正常组织", "color": "#4CAF50", "description": "正常皮肤组织", "description_en": "Normal skin tissue"},
    1: {"en": "Superficial partial-thickness", "cn": "浅表部分厚度烧伤", "color": "#FF9800", "description": "表皮和部分真皮受损", "description_en": "Epidermis and partial dermis damage"},
    2: {"en": "Deep partial-thickness", "cn": "深层部分厚度烧伤", "color": "#FF5722", "description": "真皮深层受损", "description_en": "Deep dermis damage"},
    3: {"en": "Full-thickness", "cn": "全层厚度烧伤", "color": "#F44336", "description": "皮肤全层受损", "description_en": "Full-thickness skin damage"},
    4: {"en": "Electrical", "cn": "电击烧伤", "color": "#9C27B0", "description": "电击导致的组织损伤", "description_en": "Tissue damage caused by electric shock"},
    5: {"en": "Flame", "cn": "火焰烧伤", "color": "#795548", "description": "火焰直接接触导致的烧伤", "description_en": "Burn caused by direct flame contact"}
}

# 初始化session state
if 'language' not in st.session_state: st.session_state.language = '中文'
if 'chart_colors' not in st.session_state: st.session_state.chart_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']
if 'title_font' not in st.session_state: st.session_state.title_font = {'family': 'Microsoft YaHei', 'size': 14, 'weight': 'bold'}
if 'label_font' not in st.session_state: st.session_state.label_font = {'family': 'Microsoft YaHei', 'size': 10}
if 'theme' not in st.session_state: st.session_state.theme = 'light'
if 'data_precision' not in st.session_state: st.session_state.data_precision = 10
if 'chart_title_font' not in st.session_state: st.session_state.chart_title_font = {'family': 'Microsoft YaHei', 'size': 14, 'weight': 'bold'}
if 'chart_axis_font' not in st.session_state: st.session_state.chart_axis_font = {'family': 'Microsoft YaHei', 'size': 10}
if 'chart_tick_font' not in st.session_state: st.session_state.chart_tick_font = {'family': 'Microsoft YaHei', 'size': 8}
if 'chart_label_font' not in st.session_state: st.session_state.chart_label_font = {'family': 'Microsoft YaHei', 'size': 9}

# 侧边栏
with st.sidebar:
    st.markdown("""
    <div style="text-align: center;">
        <h1>🔥👶</h1>
        <h3>儿童皮肤损伤识别系统</h3>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    app_mode = st.selectbox("选择应用模式", ["🔬 烧伤识别分析", "📖 使用指南", "⚙️ 系统设置"])
    st.markdown("---")
    if st.session_state.model is not None: 
        st.success("✅ 模型已加载")
    else: 
        st.error("❌ 模型加载失败")

# 主页面内容
if app_mode == "🔬 烧伤识别分析":
    st.markdown('<div class="main-header">🔥👶 儿童皮肤损伤智能识别与分析系统</div>', unsafe_allow_html=True)
    
    if st.session_state.model is not None:
        model = st.session_state.model
        st.success("✅ 专业模式 - 使用训练好的随机森林模型")
    else:
        st.error("❌ 模型加载失败，无法进行分析")
        st.stop()
    
    tab1, tab2 = st.tabs(["🔍 单样本分析", "📊 批量分析"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.session_state.language == '中文':
                st.markdown('<div class="sub-header">📋 输入烧伤特征参数</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sub-header">📋 Input Burn Characteristics</div>', unsafe_allow_html=True)
            
            with st.form("input_form"):
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    feature1 = st.number_input("BG1 生物标志物", value=-7.085353202, format="%.10f", help="第一个表观标志物参数")
                    feature2 = st.number_input("Ascorbic acid (抗坏血酸)", value=45874.83777, format="%.10f", help="抗坏血酸浓度")
                    feature3 = st.number_input("Pregnenolone sulfate (孕烯醇酮硫酸酯)", value=31430.32155, format="%.10f", help="孕烯醇酮硫酸酯浓度")
                    feature4 = st.number_input("IL-1β (pg/mL)", value=422.8258998, format="%.10f", help="白细胞介素-1β浓度")
                with col1_2:
                    feature5 = st.number_input("5-Methoxytryptamine (5-甲氧基色胺)", value=23673.82157, format="%.10f", help="5-甲氧基色胺浓度")
                    feature6 = st.number_input("EGF (pg/mL)", value=767.7878056, format="%.10f", help="表皮生长因子浓度")
                    feature7 = st.number_input("BG2 生物标志物", value=1.106613969, format="%.10f", help="第二个表观标志物参数")
                
                if st.session_state.language == '中文':
                    advanced_analysis = st.checkbox("执行SHAP+图网络+反事实分析", value=True)
                    submitted = st.form_submit_button("🚀 开始分析", use_container_width=True)
                else:
                    advanced_analysis = st.checkbox("Perform SHAP+Graph+Counterfactual Analysis", value=True)
                    submitted = st.form_submit_button("🚀 Start Analysis", use_container_width=True)
        
        with col2:
            if st.session_state.language == '中文':
                st.markdown('<div class="sub-header">💡 参数说明</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="feature-box"><strong>BG1:</strong> 关键生物标志物1，反映组织炎症状态</div>
                <div class="feature-box"><strong>Ascorbic acid:</strong> 抗坏血酸，抗氧化剂</div>
                <div class="feature-box"><strong>Pregnenolone sulfate:</strong> 孕烯醇酮硫酸酯，神经类固醇</div>
                <div class="feature-box"><strong>IL-1β:</strong> 炎症因子，浓度与烧伤严重程度相关</div>
                <div class="feature-box"><strong>5-Methoxytryptamine:</strong> 5-甲氧基色胺，神经递质</div>
                <div class="feature-box"><strong>EGF:</strong> 表皮生长因子，促进伤口愈合</div>
                <div class="feature-box"><strong>BG2:</strong> 关键生物标志物2，组织修复指标</div>
                """, unsafe_allow_html=True)
        
        if submitted:
            try:
                input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]], 
                                         columns=model.feature_names_in_)
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                if st.session_state.language == '中文':
                    st.markdown('<div class="sub-header">📊 分析结果</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
                
                col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
                with col_res2:
                    burn_info = burn_type_mapping[prediction]
                    if st.session_state.language == '中文':
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>诊断结果: {burn_info['cn']}</h3>
                            <p><strong>英文名称:</strong> {burn_info['en']}</p>
                            <p><strong>描述:</strong> {burn_info['description']}</p>
                            <p><strong>置信度:</strong> {probabilities[prediction]:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Diagnosis Result: {burn_info['en']}</h3>
                            <p><strong>Chinese Name:</strong> {burn_info['cn']}</p>
                            <p><strong>Description:</strong> {burn_info['description_en']}</p>
                            <p><strong>Confidence:</strong> {probabilities[prediction]:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 显示3D皮肤模型
                st.markdown("---")
                if st.session_state.language == '中文':
                    st.markdown('<div class="sub-header">🧬 3D皮肤模型 - 烧伤深度可视化</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sub-header">🧬 3D Skin Model - Burn Depth Visualization</div>', unsafe_allow_html=True)

                # 添加烧伤颜色设置
                col_color1, col_color2 = st.columns(2)
                with col_color1:
                    burn_color = st.color_picker("选择烧伤区域颜色", "#FF4500", key="burn_color_3d")
                with col_color2:
                    burn_opacity = st.slider("烧伤区域透明度", 0.1, 1.0, 0.7, 0.1, key="burn_opacity_3d")

                # 创建3D皮肤模型
                fig_3d = create_skin_3d_model_with_burn_depth(prediction, probabilities, burn_color, burn_opacity)
                st.plotly_chart(fig_3d, use_container_width=True)
                
                if advanced_analysis:
                    if st.session_state.language == '中文':
                        st.markdown('<div class="sub-header">🔬 高级模型分析</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sub-header">🔬 Advanced Model Analysis</div>', unsafe_allow_html=True)
                    
                    # SHAP分析
                    with st.spinner("正在进行SHAP分析..." if st.session_state.language == '中文' else "Performing SHAP analysis..."):
                        shap_results = perform_shap_analysis(model, input_data, model.feature_names_in_)
                    
                    # 图网络分析
                    with st.spinner("正在进行图网络分析..." if st.session_state.language == '中文' else "Performing graph network analysis..."):
                        graph_results = perform_graph_analysis([feature1, feature2, feature3, feature4, feature5, feature6, feature7], 
                                                              model.feature_names_in_, prediction, burn_type_mapping)
                    
                    # 反事实分析
                    if prediction != 0:
                        with st.spinner("正在进行反事实分析..." if st.session_state.language == '中文' else "Performing counterfactual analysis..."):
                            counterfactual_results = perform_counterfactual_analysis(model, input_data, prediction, model.feature_names_in_, burn_type_mapping)
                    else:
                        counterfactual_results = {'skip_analysis': True}
                        if st.session_state.language == '中文':
                            st.info("✅ 当前诊断为正常组织，无需进行反事实分析")
                        else:
                            st.info("✅ Current diagnosis is normal tissue, counterfactual analysis skipped")
                    
                    # 显示SHAP分析结果
                    if shap_results:
                        if st.session_state.language == '中文':
                            st.markdown("##### 📈 SHAP多类别分析")
                        else:
                            st.markdown("##### 📈 SHAP Multi-Class Analysis")
                        
                        col_shap1, col_shap2 = st.columns([1, 1])
                        
                        with col_shap1:
                            fig_combined = plot_combined_shap_analysis(shap_results, model.feature_names_in_, burn_type_mapping)
                            if fig_combined:
                                st.pyplot(fig_combined)
                                if st.session_state.language == '中文':
                                    st.caption("图1: SHAP合并分析 - 特征影响方向和重要性")
                                else:
                                    st.caption("Figure 1: Combined SHAP Analysis - Feature Impact and Importance")
                        
                        with col_shap2:
                            fig_current = plot_current_prediction_shap(shap_results, model.feature_names_in_, burn_type_mapping)
                            if fig_current:
                                st.pyplot(fig_current)
                                if st.session_state.language == '中文':
                                    st.caption("图2: 当前预测类别特征重要性分析")
                                else:
                                    st.caption("Figure 2: Feature Importance for Current Prediction")
                    
                    # 显示图网络分析结果
                    if graph_results:
                        if st.session_state.language == '中文':
                            st.markdown("##### 🔗 特征关联图网络分析")
                        else:
                            st.markdown("##### 🔗 Feature Correlation Graph Analysis")
                        
                        graph_fig = plot_optimized_graph_analysis(graph_results, model.feature_names_in_, burn_info)
                        if graph_fig:
                            st.pyplot(graph_fig)
                    
                    # 显示反事实分析结果
                    if counterfactual_results and not counterfactual_results.get('skip_analysis', False) and counterfactual_results['normal_tissue_suggestions']:
                        if st.session_state.language == '中文':
                            st.markdown("##### 🔄 反事实分析与恢复正常组织建议")
                        else:
                            st.markdown("##### 🔄 Counterfactual Analysis and Normal Tissue Recovery Suggestions")
                        
                        counterfactual_fig = plot_optimized_counterfactual_analysis(counterfactual_results, burn_type_mapping)
                        if counterfactual_fig:
                            st.pyplot(counterfactual_fig)
                        
                        if st.session_state.language == '中文':
                            st.markdown("###### 💡 恢复到正常组织的调整建议:")
                            for i, suggestion in enumerate(counterfactual_results['normal_tissue_suggestions'][:3], 1):
                                st.markdown(f"""
                                <div class="analysis-box">
                                <strong>方案 {i}:</strong> 将 <strong>{suggestion['feature']}</strong> {suggestion['change_direction']}到原来的 <strong>{suggestion['required_change']}</strong><br>
                                - 原始值: {suggestion['original_value']:.10f} → 调整后值: {suggestion['new_value']:.10f}<br>
                                - 预测置信度: {suggestion['confidence']:.2%}<br>
                                - 效果: 预测结果从 <strong>{burn_type_mapping[counterfactual_results['original_prediction']]['cn']}</strong> 恢复到 <strong>正常组织</strong>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("###### 💡 Adjustment suggestions to restore normal tissue:")
                            for i, suggestion in enumerate(counterfactual_results['normal_tissue_suggestions'][:3], 1):
                                st.markdown(f"""
                                <div class="analysis-box">
                                <strong>Scenario {i}:</strong> Change <strong>{suggestion['feature']}</strong> to <strong>{suggestion['required_change']}</strong> of original<br>
                                - Original value: {suggestion['original_value']:.10f} → Adjusted value: {suggestion['new_value']:.10f}<br>
                                - Prediction confidence: {suggestion['confidence']:.2%}<br>
                                - Effect: Prediction changes from <strong>{burn_type_mapping[counterfactual_results['original_prediction']]['en']}</strong> to <strong>Normal Tissue</strong>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # 概率分布图
                    st.markdown("---")
                    if st.session_state.language == '中文':
                        st.markdown('<div class="sub-header">📈 概率分布分析</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sub-header">📈 Probability Distribution Analysis</div>', unsafe_allow_html=True)
                    
                    font_settings = get_chart_font_settings()
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    if st.session_state.language == '中文':
                        title1, title2, ylabel = '烧伤类型概率分布', '概率分布饼图', '概率'
                        labels = [burn_type_mapping[i]['cn'] for i in range(len(probabilities))]
                    else:
                        title1, title2, ylabel = 'Burn Type Probability Distribution', 'Probability Distribution Pie Chart', 'Probability'
                        labels = [burn_type_mapping[i]['en'] for i in range(len(probabilities))]
                    
                    colors = st.session_state.chart_colors[:len(probabilities)]
                    bars = ax1.bar(range(len(probabilities)), probabilities, color=colors)
                    ax1.set_title(title1, fontfamily=font_settings['title_font']['family'], fontsize=font_settings['title_font']['size'])
                    ax1.set_xticks(range(len(probabilities)))
                    ax1.set_xticklabels(labels, rotation=45, ha='right', fontfamily=font_settings['tick_font']['family'])
                    ax1.set_ylabel(ylabel, fontfamily=font_settings['axis_font']['family'], fontsize=font_settings['axis_font']['size'])
                    ax1.set_ylim(0, 1)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.1%}', 
                                ha='center', va='bottom', fontfamily=font_settings['label_font']['family'])
                    
                    ax2.pie(probabilities, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
                           textprops={'fontfamily': font_settings['label_font']['family']})
                    ax2.set_title(title2, fontfamily=font_settings['title_font']['family'],
                                 fontsize=font_settings['title_font']['size'])
                    
                    apply_chart_font_settings(ax1, title=title1, ylabel=ylabel)
                    apply_chart_font_settings(ax2, title=title2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 结果导出
                    st.markdown("---")
                    if st.session_state.language == '中文':
                        st.markdown('<div class="sub-header">💾 结果导出</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sub-header">💾 Export Results</div>', unsafe_allow_html=True)
                    
                    # 生成增强的医疗报告
                    report_text = generate_medical_report(input_data, prediction, probabilities, shap_results, graph_results, counterfactual_results, burn_type_mapping, model.feature_names_in_, st.session_state.language)
                    
                    col_exp1, col_exp2, col_exp3 = st.columns(3)
                    with col_exp1:
                        csv_data = input_data.copy()
                        csv_data['预测类型' if st.session_state.language == '中文' else 'Predicted Type'] = burn_info['cn' if st.session_state.language == '中文' else 'en']
                        csv_data['置信度' if st.session_state.language == '中文' else 'Confidence'] = probabilities[prediction]
                        csv = csv_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 导出CSV" if st.session_state.language == '中文' else "📥 Export CSV",
                            data=csv, file_name="burn_analysis_result.csv", mime="text/csv", use_container_width=True
                        )
                    with col_exp2:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(
                            label="🖼️ 导出图表" if st.session_state.language == '中文' else "🖼️ Export Chart",
                            data=buf.getvalue(), file_name="burn_analysis_chart.png", mime="image/png", use_container_width=True
                        )
                    with col_exp3:
                        st.download_button(
                            label="📄 导出医疗报告" if st.session_state.language == '中文' else "📄 Export Medical Report",
                            data=report_text.encode('utf-8'), file_name="burn_medical_report.txt", mime="text/plain", use_container_width=True
                        )
                    
            except Exception as e:
                st.error(f"分析过程中出现错误: {str(e)}")

    with tab2:
        if st.session_state.language == '中文':
            st.markdown('<div class="sub-header">📁 批量数据处理</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sub-header">📁 Batch Data Processing</div>', unsafe_allow_html=True)
        st.info("批量分析功能开发中...")

elif app_mode == "📖 使用指南":
    st.markdown('<div class="main-header">📖 使用指南</div>', unsafe_allow_html=True)
    
    tab_guide1, tab_guide2, tab_guide3, tab_guide4, tab_guide5 = st.tabs(["📋 系统介绍", "🔬 使用步骤", "📊 数据说明", "🧠 算法原理", "❓ 常见问题"])
    with tab_guide1:
        st.markdown('<div class="guide-section">', unsafe_allow_html=True)
        st.markdown("## 🔬 系统介绍")
        st.markdown("""
        本系统基于机器学习算法，通过对生物标志物的分析，实现烧伤类型的智能识别和分类。系统集成了先进的模型可解释性技术，
        包括SHAP分析、图网络分析和反事实分析，为医疗专业人员提供全面的决策支持。
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_intro1, col_intro2 = st.columns(2)
    with col_intro1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 系统特色")
        st.markdown("""
        - **智能识别**: 基于随机森林算法的多分类模型
        - **可解释性**: 集成SHAP、图网络、反事实分析
        - **高精度**: 支持小数点后10位的数据精度
        - **可视化**: 丰富的图表和交互界面
        - **多语言**: 支持中英文界面切换
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_intro2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 📊 功能模块")
        st.markdown("""
        - **单样本分析**: 单个样本的详细分析
        - **批量分析**: 批量数据处理功能
        - **高级分析**: SHAP+图网络+反事实分析
        - **结果导出**: 支持CSV、图表、报告导出
        - **系统设置**: 个性化界面配置
        """)
        st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "⚙️ 系统设置":
    st.markdown('<div class="main-header">⚙️ 系统设置</div>', unsafe_allow_html=True)
    
    # 语言设置
    st.subheader("🌐 语言设置")
    language = st.selectbox("选择界面语言", ["中文", "English"], key="language_select")
    
    if st.button("💾 应用语言设置", use_container_width=True):
        st.session_state.language = language
        st.success("✅ 语言设置已应用")
    
    st.markdown("---")
    
    # 图表颜色设置
    st.subheader("🎨 图表颜色设置")
    st.info("当前使用Nature配色方案: #4E79A7, #F28E2B, #E15759, #76B7B2, #59A14F, #EDC948")
    
    # 应用设置按钮
    if st.button("💾 应用所有设置", use_container_width=True):
        st.success("✅ 所有设置已应用")
    
    # 重置设置为默认值
    if st.button("🔄 重置为默认设置", use_container_width=True):
        st.session_state.chart_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']
        st.session_state.title_font = {'family': 'Microsoft YaHei', 'size': 14, 'weight': 'bold'}
        st.session_state.label_font = {'family': 'Microsoft YaHei', 'size': 10}
        st.session_state.current_data_precision = 10
        st.session_state.theme = 'light'
        st.success("✅ 已重置为默认设置")
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666; font-family: "Microsoft YaHei", sans-serif;">👶 儿童皮肤损伤识别系统 | 基于机器学习的医疗辅助诊断工具 | v1.0 | 本地专业版本</div>', unsafe_allow_html=True)