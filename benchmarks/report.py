"""
报告生成模块：用于生成基准测试的数据指标文件和PDF报告
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from fpdf import FPDF
import datetime

def save_metrics_to_json(metrics, output_path="benchmark_results.json"):
    """
    将基准测试指标保存为JSON文件
    
    参数:
        metrics: 指标字典
        output_path: 输出文件路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"指标已保存到 {output_path}")

def generate_comparison_table(metrics, task_type="classification", output_path="comparison_table.csv"):
    """
    生成模型比较表格
    
    参数:
        metrics: 指标字典
        task_type: 任务类型（'classification' 或 'language_modeling'）
        output_path: 输出CSV文件路径
    """
    if task_type == "classification":
        # 分类任务的列
        columns = ["模型", "准确率", "精确率", "召回率", "F1分数", "训练时间(s)", "推理时间(s)", "参数量"]
        rows = []
        
        for model_name, model_metrics in metrics.items():
            row = [
                model_name,
                model_metrics.get("accuracy", "N/A"),
                model_metrics.get("precision", "N/A"),
                model_metrics.get("recall", "N/A"),
                model_metrics.get("f1", "N/A"),
                model_metrics.get("training_time", "N/A"),
                model_metrics.get("inference_time", "N/A"),
                model_metrics.get("parameters", "N/A")
            ]
            rows.append(row)
        
    else:  # language_modeling
        # 语言建模任务的列
        columns = ["模型", "困惑度", "损失", "训练时间(s)", "推理时间(s)", "参数量"]
        rows = []
        
        for model_name, model_metrics in metrics.items():
            row = [
                model_name,
                model_metrics.get("perplexity", "N/A"),
                model_metrics.get("loss", "N/A"),
                model_metrics.get("training_time", "N/A"),
                model_metrics.get("inference_time", "N/A"),
                model_metrics.get("parameters", "N/A")
            ]
            rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # 保存为CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"比较表格已保存到 {output_path}")
    
    return df

def plot_training_curves(metrics, output_dir="plots"):
    """
    绘制训练曲线
    
    参数:
        metrics: 指标字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 每个模型的损失曲线
    plt.figure(figsize=(12, 6))
    for model_name, model_metrics in metrics.items():
        if "train_loss" in model_metrics and "val_loss" in model_metrics:
            plt.plot(model_metrics["train_loss"], label=f"{model_name} 训练")
            plt.plot(model_metrics["val_loss"], label=f"{model_name} 验证", linestyle="--")
    
    plt.title("训练和验证损失")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    # 精度/困惑度曲线
    plt.figure(figsize=(12, 6))
    for model_name, model_metrics in metrics.items():
        if "train_accuracy" in model_metrics and "val_accuracy" in model_metrics:
            plt.plot(model_metrics["train_accuracy"], label=f"{model_name} 训练")
            plt.plot(model_metrics["val_accuracy"], label=f"{model_name} 验证", linestyle="--")
    
    # 确定正确的标题和y轴标签
    if "perplexity" in next(iter(metrics.values())):
        plt.title("训练和验证困惑度")
        plt.ylabel("困惑度")
    else:
        plt.title("训练和验证准确率")
        plt.ylabel("准确率")
    
    plt.xlabel("轮次")
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(output_dir, "accuracy_curves.png")
    plt.savefig(acc_plot_path)
    plt.close()
    
    return loss_plot_path, acc_plot_path

def plot_comparison_metrics(df, task_type="classification", output_dir="plots"):
    """
    绘制模型性能比较图
    
    参数:
        df: 包含性能指标的DataFrame
        task_type: 任务类型（'classification' 或 'language_modeling'）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []
    
    if task_type == "classification":
        # 准确率条形图
        plt.figure(figsize=(10, 6))
        sns.barplot(x="模型", y="准确率", data=df)
        plt.title("各模型准确率比较")
        plt.ylim(0, 1)
        plt.grid(True, axis="y")
        plt.tight_layout()
        acc_plot_path = os.path.join(output_dir, "accuracy_comparison.png")
        plt.savefig(acc_plot_path)
        plt.close()
        plot_paths.append(acc_plot_path)
        
        # F1分数条形图
        plt.figure(figsize=(10, 6))
        sns.barplot(x="模型", y="F1分数", data=df)
        plt.title("各模型F1分数比较")
        plt.ylim(0, 1)
        plt.grid(True, axis="y")
        plt.tight_layout()
        f1_plot_path = os.path.join(output_dir, "f1_comparison.png")
        plt.savefig(f1_plot_path)
        plt.close()
        plot_paths.append(f1_plot_path)
        
    else:  # language_modeling
        # 困惑度条形图（值越低越好）
        plt.figure(figsize=(10, 6))
        sns.barplot(x="模型", y="困惑度", data=df)
        plt.title("各模型困惑度比较（值越低越好）")
        plt.grid(True, axis="y")
        plt.tight_layout()
        ppl_plot_path = os.path.join(output_dir, "perplexity_comparison.png")
        plt.savefig(ppl_plot_path)
        plt.close()
        plot_paths.append(ppl_plot_path)
    
    # 训练时间条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x="模型", y="训练时间(s)", data=df)
    plt.title("各模型训练时间比较")
    plt.grid(True, axis="y")
    plt.tight_layout()
    train_time_plot_path = os.path.join(output_dir, "training_time_comparison.png")
    plt.savefig(train_time_plot_path)
    plt.close()
    plot_paths.append(train_time_plot_path)
    
    # 推理时间条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x="模型", y="推理时间(s)", data=df)
    plt.title("各模型推理时间比较")
    plt.grid(True, axis="y")
    plt.tight_layout()
    inference_time_plot_path = os.path.join(output_dir, "inference_time_comparison.png")
    plt.savefig(inference_time_plot_path)
    plt.close()
    plot_paths.append(inference_time_plot_path)
    
    # 参数量条形图
    if "参数量" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="模型", y="参数量", data=df)
        plt.title("各模型参数量比较")
        plt.grid(True, axis="y")
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        plt.tight_layout()
        params_plot_path = os.path.join(output_dir, "parameters_comparison.png")
        plt.savefig(params_plot_path)
        plt.close()
        plot_paths.append(params_plot_path)
    
    return plot_paths

class PDFReport(FPDF):
    """生成PDF报告的类"""
    
    def __init__(self, title="模型基准测试报告", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.add_font('simhei', '', '/System/Library/Fonts/STHeiti Medium.ttc', uni=True)
        
    def header(self):
        # 设置字体和大小
        self.set_font('simhei', '', 15)
        # 标题
        self.cell(0, 10, self.title, 0, 1, 'C')
        # 设置生成日期
        self.set_font('simhei', '', 10)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 5, f"生成日期：{date_str}", 0, 1, 'R')
        # 水平线
        self.line(10, 25, 200, 25)
        # 空行
        self.ln(10)
        
    def footer(self):
        # 页脚定位
        self.set_y(-15)
        # 设置字体
        self.set_font('simhei', '', 8)
        # 页码
        self.cell(0, 10, f'第 {self.page_no()} 页', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('simhei', '', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
    
    def chapter_body(self, text):
        self.set_font('simhei', '', 12)
        self.multi_cell(0, 6, text)
        self.ln()
    
    def add_table(self, df, title):
        self.chapter_title(title)
        
        # 生成表格内容
        table_data = [df.columns.tolist()] + df.values.tolist()
        table_str = tabulate(table_data, headers="firstrow", tablefmt="grid")
        
        # 使用monospaced字体展示表格
        self.set_font('simhei', '', 10)
        self.multi_cell(0, 5, table_str)
        self.ln(10)
    
    def add_image(self, img_path, title, w=190):
        self.chapter_title(title)
        
        # 计算图像高度，保持宽高比
        img_w, img_h = self.get_image_size(img_path)
        h = w * img_h / img_w
        
        # 居中展示图像
        x = (210 - w) / 2
        self.image(img_path, x=x, w=w, h=h)
        self.ln(10)
    
    def get_image_size(self, img_path):
        """获取图像大小并返回宽度和高度"""
        try:
            from PIL import Image
            img = Image.open(img_path)
            return img.width, img.height
        except:
            # 如果无法使用PIL，返回默认尺寸
            return 800, 600

def generate_pdf_report(
    metrics, 
    task_type="classification", 
    output_path="benchmark_report.pdf",
    dataset_name="SST-2"
):
    """
    生成PDF基准测试报告
    
    参数:
        metrics: 指标字典
        task_type: 任务类型（'classification' 或 'language_modeling'）
        output_path: 输出PDF文件路径
        dataset_name: 数据集名称
    """
    # 生成比较表格和图表
    metrics_df = generate_comparison_table(metrics, task_type)
    loss_plot, acc_plot = plot_training_curves(metrics)
    comparison_plots = plot_comparison_metrics(metrics_df, task_type)
    
    # 创建PDF报告
    report_title = f"QCIT+与其他模型基准测试报告：{dataset_name}数据集"
    pdf = PDFReport(title=report_title)
    
    # 添加摘要
    pdf.chapter_title("摘要")
    summary = (
        f"本报告对QCIT+模型与BERT、Standard和GPT-2在{dataset_name}数据集上的性能进行了比较。"
        f"评估包括各模型在{'情感分类' if task_type == 'classification' else '语言建模'}任务上的"
        f"{'准确率、精确率、召回率和F1分数' if task_type == 'classification' else '困惑度和损失'}，"
        "以及训练时间、推理时间和参数量。"
    )
    pdf.chapter_body(summary)
    
    # 添加比较表格
    pdf.add_page()
    pdf.add_table(metrics_df, "模型性能比较")
    
    # 添加训练曲线
    pdf.add_page()
    pdf.add_image(loss_plot, "训练和验证损失曲线")
    
    pdf.add_page()
    if task_type == "classification":
        pdf.add_image(acc_plot, "训练和验证准确率曲线")
    else:
        pdf.add_image(acc_plot, "训练和验证困惑度曲线")
    
    # 添加比较图表
    for plot_path in comparison_plots:
        pdf.add_page()
        title = os.path.basename(plot_path).replace("_", " ").replace(".png", "").capitalize()
        pdf.add_image(plot_path, title)
    
    # 添加结论
    pdf.add_page()
    pdf.chapter_title("结论与讨论")
    
    # 找出最佳模型
    if task_type == "classification":
        best_model = metrics_df.loc[metrics_df["准确率"].idxmax()]["模型"]
        best_metric = metrics_df["准确率"].max()
        metric_name = "准确率"
    else:
        best_model = metrics_df.loc[metrics_df["困惑度"].idxmin()]["模型"]
        best_metric = metrics_df["困惑度"].min()
        metric_name = "困惑度"
    
    fastest_train = metrics_df.loc[metrics_df["训练时间(s)"].idxmin()]["模型"]
    fastest_inference = metrics_df.loc[metrics_df["推理时间(s)"].idxmin()]["模型"]
    
    conclusion = (
        f"在{dataset_name}数据集上的测试结果表明，{best_model}模型在{metric_name}指标上表现最佳"
        f"({best_metric:.4f})。{fastest_train}模型训练速度最快，"
        f"而{fastest_inference}模型推理速度最快。\n\n"
        "QCIT+模型的特性及其在此基准测试中的表现显示了量子经典同构变换器在处理"
        f"{'分类' if task_type == 'classification' else '语言建模'}任务时的优势和局限性。"
        "后续研究可以尝试优化模型结构和训练策略，进一步提升QCIT+模型的性能。"
    )
    pdf.chapter_body(conclusion)
    
    # 保存PDF
    pdf.output(output_path)
    print(f"PDF报告已生成：{output_path}")
    
    return output_path 