import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class PerformanceMonitor:
    """性能监控和报告生成器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统性能指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used / (1024**3)  # GB
            memory_total = memory.total / (1024**3)  # GB
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used = disk.used / (1024**3)  # GB
            disk_total = disk.total / (1024**3)  # GB
            
            # 网络IO
            net_io = psutil.net_io_counters()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'usage_percent': cpu_percent,
                    'core_count': cpu_count
                },
                'memory': {
                    'usage_percent': memory_percent,
                    'used_gb': round(memory_used, 2),
                    'total_gb': round(memory_total, 2)
                },
                'disk': {
                    'usage_percent': round(disk_percent, 2),
                    'used_gb': round(disk_used, 2),
                    'total_gb': round(disk_total, 2)
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            }
        except Exception as e:
            self.logger.error(f"收集系统指标时出错: {e}")
            return {}
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """收集应用程序性能指标"""
        try:
            current_process = psutil.Process()
            
            # 应用程序CPU和内存使用
            app_cpu = current_process.cpu_percent()
            app_memory = current_process.memory_info()
            
            # 运行时间
            uptime = time.time() - self.start_time
            
            return {
                'uptime_seconds': round(uptime, 2),
                'uptime_formatted': str(timedelta(seconds=int(uptime))),
                'cpu_percent': app_cpu,
                'memory_mb': round(app_memory.rss / (1024**2), 2),
                'threads_count': current_process.num_threads(),
                'open_files': len(current_process.open_files()) if hasattr(current_process, 'open_files') else 0
            }
        except Exception as e:
            self.logger.error(f"收集应用指标时出错: {e}")
            return {}
    
    def calculate_performance_trends(self) -> Dict[str, Any]:
        """计算性能趋势分析"""
        if len(self.metrics_history) < 2:
            return {'trend_analysis': '数据不足，无法分析趋势'}
        
        try:
            recent_metrics = self.metrics_history[-10:]  # 最近10次记录
            
            # CPU趋势
            cpu_values = [m.get('system', {}).get('cpu', {}).get('usage_percent', 0) for m in recent_metrics]
            cpu_trend = 'stable'
            if len(cpu_values) >= 3:
                if cpu_values[-1] > cpu_values[-3] + 10:
                    cpu_trend = 'increasing'
                elif cpu_values[-1] < cpu_values[-3] - 10:
                    cpu_trend = 'decreasing'
            
            # 内存趋势
            memory_values = [m.get('system', {}).get('memory', {}).get('usage_percent', 0) for m in recent_metrics]
            memory_trend = 'stable'
            if len(memory_values) >= 3:
                if memory_values[-1] > memory_values[-3] + 5:
                    memory_trend = 'increasing'
                elif memory_values[-1] < memory_values[-3] - 5:
                    memory_trend = 'decreasing'
            
            return {
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'avg_cpu': round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else 0,
                'avg_memory': round(sum(memory_values) / len(memory_values), 2) if memory_values else 0,
                'samples_count': len(recent_metrics)
            }
        except Exception as e:
            self.logger.error(f"计算性能趋势时出错: {e}")
            return {'error': str(e)}
    
    def get_performance_alerts(self, system_metrics: Dict, app_metrics: Dict) -> List[Dict[str, str]]:
        """生成性能警告"""
        alerts = []
        
        try:
            # CPU警告
            cpu_usage = system_metrics.get('cpu', {}).get('usage_percent', 0)
            if cpu_usage > 80:
                alerts.append({
                    'level': 'warning' if cpu_usage < 90 else 'critical',
                    'type': 'cpu',
                    'message': f'CPU使用率过高: {cpu_usage}%'
                })
            
            # 内存警告
            memory_usage = system_metrics.get('memory', {}).get('usage_percent', 0)
            if memory_usage > 85:
                alerts.append({
                    'level': 'warning' if memory_usage < 95 else 'critical',
                    'type': 'memory',
                    'message': f'内存使用率过高: {memory_usage}%'
                })
            
            # 磁盘警告
            disk_usage = system_metrics.get('disk', {}).get('usage_percent', 0)
            if disk_usage > 90:
                alerts.append({
                    'level': 'warning' if disk_usage < 95 else 'critical',
                    'type': 'disk',
                    'message': f'磁盘使用率过高: {disk_usage}%'
                })
            
            # 应用程序内存警告
            app_memory = app_metrics.get('memory_mb', 0)
            if app_memory > 1000:  # 1GB
                alerts.append({
                    'level': 'warning',
                    'type': 'app_memory',
                    'message': f'应用程序内存使用过高: {app_memory}MB'
                })
                
        except Exception as e:
            self.logger.error(f"生成性能警告时出错: {e}")
            alerts.append({
                'level': 'error',
                'type': 'system',
                'message': f'警告生成失败: {str(e)}'
            })
        
        return alerts
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        生成详细的性能报告
        
        Returns:
            Dict: 包含完整性能分析数据的字典
        """
        try:
            # 收集当前性能指标
            system_metrics = self.collect_system_metrics()
            app_metrics = self.collect_application_metrics()
            
            # 保存到历史记录
            current_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'system': system_metrics,
                'application': app_metrics
            }
            self.metrics_history.append(current_snapshot)
            
            # 保持历史记录在合理范围内（最多保留100条记录）
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            # 计算趋势分析
            trends = self.calculate_performance_trends()
            
            # 生成警告
            alerts = self.get_performance_alerts(system_metrics, app_metrics)
            
            # 生成性能评分
            performance_score = self._calculate_performance_score(system_metrics, app_metrics)
            
            # 构建完整报告
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_version': '2.0',
                    'monitoring_duration': app_metrics.get('uptime_formatted', 'Unknown')
                },
                'current_performance': {
                    'system_metrics': system_metrics,
                    'application_metrics': app_metrics,
                    'performance_score': performance_score
                },
                'trend_analysis': trends,
                'alerts': alerts,
                'recommendations': self._generate_recommendations(system_metrics, app_metrics, alerts),
                'summary': self._generate_summary(system_metrics, app_metrics, performance_score, len(alerts))
            }
            
            self.logger.info("性能报告生成成功")
            return report
            
        except Exception as e:
            error_msg = f"生成性能报告时发生错误: {str(e)}"
            self.logger.error(error_msg)
            
            # 返回错误报告而不是抛出异常
            return {
                'error': True,
                'error_message': error_msg,
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_version': '2.0',
                    'status': 'failed'
                },
                'fallback_data': {
                    'basic_info': 'Performance monitoring encountered an error',
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _calculate_performance_score(self, system_metrics: Dict, app_metrics: Dict) -> Dict[str, Any]:
        """计算性能评分"""
        try:
            scores = {}
            
            # CPU评分 (0-100)
            cpu_usage = system_metrics.get('cpu', {}).get('usage_percent', 0)
            cpu_score = max(0, 100 - cpu_usage)
            scores['cpu'] = round(cpu_score, 1)
            
            # 内存评分 (0-100)
            memory_usage = system_metrics.get('memory', {}).get('usage_percent', 0)
            memory_score = max(0, 100 - memory_usage)
            scores['memory'] = round(memory_score, 1)
            
            # 磁盘评分 (0-100)
            disk_usage = system_metrics.get('disk', {}).get('usage_percent', 0)
            disk_score = max(0, 100 - disk_usage)
            scores['disk'] = round(disk_score, 1)
            
            # 总体评分
            overall_score = (cpu_score + memory_score + disk_score) / 3
            scores['overall'] = round(overall_score, 1)
            
            # 性能等级
            if overall_score >= 80:
                performance_grade = 'Excellent'
            elif overall_score >= 60:
                performance_grade = 'Good'
            elif overall_score >= 40:
                performance_grade = 'Fair'
            else:
                performance_grade = 'Poor'
            
            scores['grade'] = performance_grade
            
            return scores
            
        except Exception as e:
            self.logger.error(f"计算性能评分时出错: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, system_metrics: Dict, app_metrics: Dict, alerts: List) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        try:
            # 基于CPU使用率的建议
            cpu_usage = system_metrics.get('cpu', {}).get('usage_percent', 0)
            if cpu_usage > 80:
                recommendations.append("考虑优化CPU密集型操作或增加CPU资源")
            
            # 基于内存使用率的建议
            memory_usage = system_metrics.get('memory', {}).get('usage_percent', 0)
            if memory_usage > 85:
                recommendations.append("建议释放不必要的内存或增加内存容量")
            
            # 基于磁盘使用率的建议
            disk_usage = system_metrics.get('disk', {}).get('usage_percent', 0)
            if disk_usage > 90:
                recommendations.append("磁盘空间不足，建议清理临时文件或扩展存储")
            
            # 基于应用程序指标的建议
            app_memory = app_metrics.get('memory_mb', 0)
            if app_memory > 1000:
                recommendations.append("应用程序内存使用较高，检查是否存在内存泄漏")
            
            # 基于警告的建议
            critical_alerts = [alert for alert in alerts if alert.get('level') == 'critical']
            if critical_alerts:
                recommendations.append("存在严重性能问题，建议立即处理关键警告")
            
            if not recommendations:
                recommendations.append("系统性能良好，继续保持当前配置")
                
        except Exception as e:
            self.logger.error(f"生成建议时出错: {e}")
            recommendations.append("建议生成失败，请检查系统状态")
        
        return recommendations
    
    def _generate_summary(self, system_metrics: Dict, app_metrics: Dict, performance_score: Dict, alert_count: int) -> str:
        """生成性能摘要"""
        try:
            cpu_usage = system_metrics.get('cpu', {}).get('usage_percent', 0)
            memory_usage = system_metrics.get('memory', {}).get('usage_percent', 0)
            overall_score = performance_score.get('overall', 0)
            grade = performance_score.get('grade', 'Unknown')
            uptime = app_metrics.get('uptime_formatted', 'Unknown')
            
            summary = f"系统运行时间: {uptime}，整体性能评分: {overall_score}/100 ({grade})。"
            summary += f"当前CPU使用率: {cpu_usage}%，内存使用率: {memory_usage}%。"
            
            if alert_count > 0:
                summary += f"检测到 {alert_count} 个性能警告，建议及时处理。"
            else:
                summary += "系统运行正常，无性能警告。"
                
            return summary
            
        except Exception as e:
            self.logger.error(f"生成摘要时出错: {e}")
            return "性能摘要生成失败"

# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建性能监控实例
    monitor = PerformanceMonitor()
    
    # 生成性能报告
    report = monitor.get_performance_report()
    
    # 输出报告（格式化JSON）
    print(json.dumps(report, indent=2, ensure_ascii=False))