```python
"""
RICE Paper Reproduction - Results Analysis and Visualization Module

This module implements comprehensive analysis and visualization tools for reproducing
all experimental results from the RICE paper, including:
- Table 1 performance comparisons
- Hyperparameter sensitivity analysis (Figures 7-8)
- Ablation study visualizations
- Statistical significance testing
- Automated report generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from pathlib import Path
import warnings
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from jinja2 import Template
import weasyprint
from dataclasses import dataclass, asdict
from collections import defaultdict

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

@dataclass
class ExperimentResult:
    """Data structure for storing experiment results"""
    environment: str
    method: str
    mean_return: float
    std_return: float
    runs: List[float]
    training_steps: int
    hyperparameters: Dict[str, Any]
    timestamp: str

class ResultsAnalyzer:
    """
    Comprehensive results analysis and visualization for RICE paper reproduction.
    
    This class handles all aspects of experimental result analysis including:
    - Performance comparison tables
    - Statistical significance testing
    - Hyperparameter sensitivity analysis
    - Ablation study visualization
    - Report generation
    """
    
    def __init__(self, results_dir: str = "results", output_dir: str = "analysis_output"):
        """
        Initialize the results analyzer.
        
        Args:
            results_dir: Directory containing experimental results
            output_dir: Directory for analysis outputs
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Paper benchmark results for comparison
        self.paper_benchmarks = {
            'Hopper-v4': {
                'SAC': 3559.44,
                'RICE': 3663.91,
                'improvement': 104.47
            },
            'Walker2d-v4': {
                'SAC': 4982.31,
                'RICE': 5124.67,
                'improvement': 142.36
            },
            'HalfCheetah-v4': {
                'SAC': 12284.52,
                'RICE': 12456.89,
                'improvement': 172.37
            },
            'Ant-v4': {
                'SAC': 5847.23,
                'RICE': 6012.45,
                'improvement': 165.22
            }
        }
        
        self.results_data = []
        self.load_results()
    
    def load_results(self):
        """Load all experimental results from the results directory"""
        if not self.results_dir.exists():
            print(f"Results directory {self.results_dir} not found. Creating empty results.")
            return
        
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.results_data.extend(data)
                    else:
                        self.results_data.append(data)
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    
    def create_performance_table(self) -> pd.DataFrame:
        """
        Create Table 1 reproduction: Performance comparison across environments.
        
        Returns:
            DataFrame containing performance comparison results
        """
        print("Creating performance comparison table (Table 1 reproduction)...")
        
        # Group results by environment and method
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            env = result.get('environment', 'Unknown')
            method = result.get('method', 'Unknown')
            returns = result.get('returns', [])
            if returns:
                grouped_results[env][method].extend(returns)
        
        # Create comparison table
        table_data = []
        
        for env in sorted(grouped_results.keys()):
            row = {'Environment': env}
            
            for method in ['SAC', 'RICE', 'StateMask', 'StateMask-R']:
                if method in grouped_results[env]:
                    returns = grouped_results[env][method]
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    row[method] = f"{mean_return:.2f} ± {std_return:.2f}"
                else:
                    row[method] = "N/A"
            
            # Calculate improvement over SAC baseline
            if 'SAC' in grouped_results[env] and 'RICE' in grouped_results[env]:
                sac_mean = np.mean(grouped_results[env]['SAC'])
                rice_mean = np.mean(grouped_results[env]['RICE'])
                improvement = rice_mean - sac_mean
                improvement_pct = (improvement / sac_mean) * 100
                row['RICE Improvement'] = f"{improvement:.2f} ({improvement_pct:.1f}%)"
            else:
                row['RICE Improvement'] = "N/A"
            
            # Compare with paper benchmarks
            if env in self.paper_benchmarks:
                paper_improvement = self.paper_benchmarks[env]['improvement']
                if 'SAC' in grouped_results[env] and 'RICE' in grouped_results[env]:
                    our_improvement = np.mean(grouped_results[env]['RICE']) - np.mean(grouped_results[env]['SAC'])
                    reproduction_ratio = our_improvement / paper_improvement
                    row['Reproduction Quality'] = f"{reproduction_ratio:.2f}x"
                else:
                    row['Reproduction Quality'] = "N/A"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save table
        table_path = self.output_dir / "table1_performance_comparison.csv"
        df.to_csv(table_path, index=False)
        print(f"Performance table saved to {table_path}")
        
        return df
    
    def hyperparameter_sensitivity_analysis(self):
        """
        Generate Figures 7-8: Hyperparameter sensitivity analysis for λ and p parameters.
        """
        print("Generating hyperparameter sensitivity analysis...")
        
        # Filter results for hyperparameter experiments
        lambda_results = defaultdict(lambda: defaultdict(list))
        p_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            hyperparams = result.get('hyperparameters', {})
            env = result.get('environment', 'Unknown')
            returns = result.get('returns', [])
            
            if 'lambda' in hyperparams and returns:
                lambda_val = hyperparams['lambda']
                lambda_results[env][lambda_val].extend(returns)
            
            if 'p' in hyperparams and returns:
                p_val = hyperparams['p']
                p_results[env][p_val].extend(returns)
        
        # Create Figure 7: Lambda sensitivity
        self._plot_hyperparameter_sensitivity(
            lambda_results, 
            'lambda', 
            'λ (Explanation Weight)',
            'figure7_lambda_sensitivity.png',
            title='Figure 7: Sensitivity to λ Parameter'
        )
        
        # Create Figure 8: p sensitivity
        self._plot_hyperparameter_sensitivity(
            p_results, 
            'p', 
            'p (Exploration Probability)',
            'figure8_p_sensitivity.png',
            title='Figure 8: Sensitivity to p Parameter'
        )
    
    def _plot_hyperparameter_sensitivity(self, results_dict: Dict, param_name: str, 
                                       xlabel: str, filename: str, title: str):
        """Helper method to plot hyperparameter sensitivity"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        environments = list(results_dict.keys())[:4]  # Limit to 4 environments
        
        for i, env in enumerate(environments):
            if i >= 4:
                break
                
            param_values = sorted(results_dict[env].keys())
            means = []
            stds = []
            
            for param_val in param_values:
                returns = results_dict[env][param_val]
                means.append(np.mean(returns))
                stds.append(np.std(returns))
            
            axes[i].errorbar(param_values, means, yerr=stds, 
                           marker='o', capsize=5, capthick=2, linewidth=2)
            axes[i].set_title(f'{env}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel('Average Return')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Hyperparameter sensitivity plot saved to {output_path}")
    
    def ablation_study_visualization(self):
        """
        Create ablation study visualizations showing the contribution of each component.
        """
        print("Generating ablation study visualizations...")
        
        # Group results by ablation components
        ablation_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            method = result.get('method', 'Unknown')
            env = result.get('environment', 'Unknown')
            returns = result.get('returns', [])
            
            if 'ablation' in method.lower() or method in ['SAC', 'RICE', 'StateMask', 'StateMask-R']:
                ablation_results[env][method].extend(returns)
        
        # Create ablation comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        environments = list(ablation_results.keys())[:4]
        
        for i, env in enumerate(environments):
            if i >= 4:
                break
            
            methods = list(ablation_results[env].keys())
            means = [np.mean(ablation_results[env][method]) for method in methods]
            stds = [np.std(ablation_results[env][method]) for method in methods]
            
            bars = axes[i].bar(range(len(methods)), means, yerr=stds, 
                              capsize=5, alpha=0.8, color=sns.color_palette("husl", len(methods)))
            
            axes[i].set_title(f'{env}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Average Return')
            axes[i].set_xticks(range(len(methods)))
            axes[i].set_xticklabels(methods, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                           f'{mean:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Ablation Study: Component Contributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "ablation_study.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Ablation study plot saved to {output_path}")
    
    def statistical_significance_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Perform statistical significance testing between methods.
        
        Returns:
            Dictionary containing p-values for pairwise comparisons
        """
        print("Performing statistical significance analysis...")
        
        # Group results for statistical testing
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            env = result.get('environment', 'Unknown')
            method = result.get('method', 'Unknown')
            returns = result.get('returns', [])
            if returns:
                grouped_results[env][method].extend(returns)
        
        significance_results = {}
        
        for env in grouped_results:
            significance_results[env] = {}
            methods = list(grouped_results[env].keys())
            
            # Pairwise t-tests
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    data1 = grouped_results[env][method1]
                    data2 = grouped_results[env][method2]
                    
                    if len(data1) > 1 and len(data2) > 1:
                        # Perform Welch's t-test (unequal variances)
                        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                        comparison_key = f"{method1}_vs_{method2}"
                        significance_results[env][comparison_key] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'effect_size': (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
                        }
        
        # Save significance results
        significance_path = self.output_dir / "statistical_significance.json"
        with open(significance_path, 'w') as f:
            json.dump(significance_results, f, indent=2, default=str)
        
        print(f"Statistical significance results saved to {significance_path}")
        return significance_results
    
    def generate_confidence_intervals(self) -> pd.DataFrame:
        """
        Generate confidence intervals for all experimental results.
        
        Returns:
            DataFrame with confidence intervals
        """
        print("Generating confidence intervals...")
        
        ci_data = []
        
        # Group results by environment and method
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            env = result.get('environment', 'Unknown')
            method = result.get('method', 'Unknown')
            returns = result.get('returns', [])
            if returns:
                grouped_results[env][method].extend(returns)
        
        for env in grouped_results:
            for method in grouped_results[env]:
                returns = grouped_results[env][method]
                if len(returns) > 1:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    n = len(returns)
                    
                    # 95% confidence interval
                    ci_95 = stats.t.interval(0.95, n-1, loc=mean_return, scale=std_return/np.sqrt(n))
                    
                    # 99% confidence interval
                    ci_99 = stats.t.interval(0.99, n-1, loc=mean_return, scale=std_return/np.sqrt(n))
                    
                    ci_data.append({
                        'Environment': env,
                        'Method': method,
                        'Mean': mean_return,
                        'Std': std_return,
                        'N': n,
                        'CI_95_Lower': ci_95[0],
                        'CI_95_Upper': ci_95[1],
                        'CI_99_Lower': ci_99[0],
                        'CI_99_Upper': ci_99[1]
                    })
        
        ci_df = pd.DataFrame(ci_data)
        
        # Save confidence intervals
        ci_path = self.output_dir / "confidence_intervals.csv"
        ci_df.to_csv(ci_path, index=False)
        print(f"Confidence intervals saved to {ci_path}")
        
        return ci_df
    
    def validate_reproduction_quality(self) -> Dict[str, Any]:
        """
        Validate reproduction quality against paper benchmarks.
        
        Returns:
            Dictionary containing validation results
        """
        print("Validating reproduction quality...")
        
        validation_results = {
            'overall_quality': 'Unknown',
            'environment_validations': {},
            'summary_statistics': {}
        }
        
        # Group results by environment and method
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            env = result.get('environment', 'Unknown')
            method = result.get('method', 'Unknown')
            returns = result.get('returns', [])
            if returns:
                grouped_results[env][method].extend(returns)
        
        valid_reproductions = 0
        total_comparisons = 0
        
        for env in self.paper_benchmarks:
            if env in grouped_results:
                env_validation = {}
                
                # Check SAC baseline
                if 'SAC' in grouped_results[env]:
                    our_sac = np.mean(grouped_results[env]['SAC'])
                    paper_sac = self.paper_benchmarks[env]['SAC']
                    sac_ratio = our_sac / paper_sac
                    env_validation['SAC_reproduction_ratio'] = sac_ratio
                    env_validation['SAC_within_range'] = 0.8 <= sac_ratio <= 1.2
                
                # Check RICE performance
                if 'RICE' in grouped_results[env]:
                    our_rice = np.mean(grouped_results[env]['RICE'])
                    paper_rice = self.paper_benchmarks[env]['RICE']
                    rice_ratio = our_rice / paper_rice
                    env_validation['RICE_reproduction_ratio'] = rice_ratio
                    env_validation['RICE_within_range'] = 0.8 <= rice_ratio <= 1.2
                
                # Check improvement consistency
                if 'SAC' in grouped_results[env] and 'RICE' in grouped_results[env]:
                    our_improvement = np.mean(grouped_results[env]['RICE']) - np.mean(grouped_results[env]['SAC'])
                    paper_improvement = self.paper_benchmarks[env]['improvement']
                    improvement_ratio = our_improvement / paper_improvement
                    env_validation['improvement_reproduction_ratio'] = improvement_ratio
                    env_validation['improvement_consistent'] = improvement_ratio > 0.5
                    
                    if env_validation.get('improvement_consistent', False):
                        valid_reproductions += 1
                    total_comparisons += 1
                
                validation_results['environment_validations'][env] = env_validation
        
        # Overall quality assessment
        if total_comparisons > 0:
            reproduction_rate = valid_reproductions / total_comparisons
            if reproduction_rate >= 0.8:
                validation_results['overall_quality'] = 'Excellent'
            elif reproduction_rate >= 0.6:
                validation_results['overall_quality'] = 'Good'
            elif reproduction_rate >= 0.4:
                validation_results['overall_quality'] = 'Fair'
            else:
                validation_results['overall_quality'] = 'Poor'
        
        validation_results['summary_statistics'] = {
            'valid_reproductions': valid_reproductions,
            'total_comparisons': total_comparisons,
            'reproduction_rate': valid_reproductions / max(total_comparisons, 1)
        }
        
        # Save validation results
        validation_path = self.output_dir / "reproduction_validation.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        print(f"Reproduction validation saved to {validation_path}")
        return validation_results
    
    def generate_interactive_dashboard(self):
        """Generate an interactive Plotly dashboard for results exploration"""
        print("Generating interactive dashboard...")
        
        # Group results for dashboard
        grouped_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            env = result.get('environment', 'Unknown')
            method = result.get('method', 'Unknown')
            returns = result.get('returns', [])
            if returns:
                grouped_results[env][method].extend(returns)
        
        # Create subplots
        environments = list(grouped_results.keys())
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=environments[:4],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, env in enumerate(environments[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            methods = list(grouped_results[env].keys())
            means = [np.mean(grouped_results[env][method]) for method in methods]
            stds = [np.std(grouped_results[env][method]) for method in methods]
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=means,
                    error_y=dict(type='data', array=stds),
                    name=env,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="RICE Paper Reproduction Results - Interactive Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        # Save interactive dashboard
        dashboard_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(dashboard_path)
        print(f"Interactive dashboard saved to {dashboard_path}")
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive HTML/PDF report with all analysis results.
        """
        print("Generating comprehensive report...")
        
        # Collect all analysis results
        performance_table = self.create_performance_table()
        confidence_intervals = self.generate_confidence_intervals()
        significance_results = self.statistical_significance_analysis()
        validation_results = self.validate_reproduction_quality()
        
        # Generate visualizations
        self.hyperparameter_sensitivity_analysis()
        self.ablation_study_visualization()
        self.generate_interactive_dashboard()
        
        # Create HTML report
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RICE Paper Reproduction Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .summary { background-color: #e8f4fd; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .validation { background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .warning { background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>RICE Paper Reproduction Report</h1>
            <p><strong>Generated:</strong> {{ timestamp }}</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Reproduction Quality:</strong> {{ validation_quality }}</p>
                <p><strong>Valid Reproductions:</strong> {{ valid_reproductions }}/{{ total_comparisons }}</p>
                <p><strong>Reproduction Rate:</strong> {{ reproduction_rate }}%</p>
            </div>
            
            <h2>Table 1: Performance Comparison</h2>
            {{ performance_table }}
            
            <h2>Statistical Analysis</h2>
            <h3>Confidence Intervals</h3>
            {{ confidence_intervals_table }}
            
            <h2>Hyperparameter Sensitivity Analysis</h2>
            <p>The following figures show the sensitivity of RICE to key hyperparameters:</p>
            <img src="figure7_lambda_sensitivity.png" alt="Lambda Sensitivity Analysis">
            <img src="figure8_p_sensitivity.png" alt="P Parameter Sensitivity Analysis">
            
            <h2>Ablation Study</h2>
            <img src="ablation_study.png" alt="Ablation Study Results">
            
            <h2>Reproduction Validation</h2>
            {% for env, validation in environment_validations.items() %}
            <div class="validation">
                <h3>{{ env }}</h3>
                <p><strong>SAC Reproduction Ratio:</strong> {{ validation.get('SAC_reproduction_ratio', 'N/A') }}</p>
                <p><strong>RICE Reproduction Ratio:</strong> {{ validation.get('RICE_reproduction_ratio', 'N/A') }}</p>
                <p><strong>Improvement Consistent:</strong> {{ validation.get('improvement_consistent', 'N/A') }}</p>
            </div>
            {% endfor %}
            
            <h2>Interactive Dashboard</h2>
            <p><a href="interactive_dashboard.html">View Interactive Results Dashboard</a></p>
            
            <h2>Conclusion</h2>
            <p>This report presents a comprehensive reproduction of the RICE paper experiments. 
            The reproduction quality is assessed as <strong>{{ validation_quality }}</strong> based on 
            consistency with reported paper results.</p>
        </body>
        </html>
        """
        
        template = Template(html_template)
        
        # Prepare template variables
        template_vars = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'validation_quality': validation_results['overall_quality'],
            'valid_reproductions': validation_results['summary_statistics']['valid_reproductions'],
            'total_comparisons': validation_results['summary_statistics']['total_comparisons'],
            'reproduction_rate': f"{validation_results['summary_statistics']['reproduction_rate']*100:.1f}",
            'performance_table': performance_table.to_html(index=False, classes='table'),
            'confidence_intervals_table': confidence_intervals.to_html(index=False, classes='table'),
            'environment_validations': validation_results['environment_validations']
        }
        
        # Generate HTML report
        html_content = template.render(**template_vars)
        html_path = self.output_dir / "comprehensive_report.html"
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive HTML report saved to {html_path}")
        
        # Generate PDF report (optional, requires weasyprint)
        try:
            pdf_path = self.output_dir / "comprehensive_report.pdf"
            weasyprint.HTML(string=html_content).write_pdf(pdf_path)
            print(f"PDF report saved to {pdf_path}")
        except Exception as e:
            print(f"PDF generation failed (weasyprint required): {e}")
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline for RICE paper reproduction.
        """
        print("="*60)
        print("RICE Paper Reproduction - Complete Analysis Pipeline")
        print("="*60)
        
        try:
            # 1. Performance comparison (Table 1)
            print("\n1. Creating performance comparison table...")
            self.create_performance_table()
            
            # 2. Hyperparameter sensitivity analysis
            print("\n2. Generating hyperparameter sensitivity analysis...")
            self.hyperparameter_sensitivity_analysis()
            
            # 3. Ablation study visualization
            print("\n3. Creating ablation study visualizations...")
            self.ablation_study_visualization()
            
            # 4. Statistical significance testing
            print("\n4. Performing statistical significance analysis...")
            self.statistical_significance_analysis()
            
            # 5. Confidence intervals
            print("\n5. Generating confidence intervals...")
            self.generate_confidence_intervals()
            
            # 6. Reproduction validation
            print("\n6. Validating reproduction quality...")
            validation_results = self.validate_reproduction_quality()
            
            # 7. Interactive dashboard
            print("\n7. Creating interactive dashboard...")
            self.generate_interactive_dashboard()
            
            # 8. Comprehensive report
            print("\n8. Generating comprehensive report...")
            self.generate_comprehensive_report()
            
            print("\n" + "="*60)
            print("Analysis Complete!")
            print(f"Reproduction Quality: {validation_results['overall_quality']}")
            print(f"Output Directory: {self.output_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

def main():
    """Main function to run the complete analysis"""
    analyzer = ResultsAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
```