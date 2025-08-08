#!/usr/bin/env python3
"""
文本提取脚本
功能：从JSON文件中提取所有文本内容，支持批量处理多个文件
包括：正文、图片说明、公式文本、表格内容等
优化：去除空行和多余的空白字符，节省LLM token消耗
作者：AI助手
"""

import json
import re
from html import unescape
from pathlib import Path
import sys


def clean_text(text):
    """
    清理文本内容，去除多余的空白字符
    
    Args:
        text (str): 原始文本
        
    Returns:
        str: 清理后的文本
    """
    if not text:
        return ""
    
    # 去除首尾空白字符
    text = text.strip()
    
    # 将多个连续的空白字符（包括空格、制表符、换行符）替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    return text


def clean_html_table(html_content):
    """
    清理HTML表格内容，提取纯文本
    
    Args:
        html_content (str): HTML格式的表格内容
        
    Returns:
        str: 清理后的纯文本
    """
    if not html_content:
        return ""
    
    # 移除HTML标签
    clean_text_content = re.sub(r'<[^>]+>', ' ', html_content)
    # 处理HTML实体
    clean_text_content = unescape(clean_text_content)
    # 清理多余的空白字符
    clean_text_content = clean_text(clean_text_content)
    
    return clean_text_content


def extract_text_from_json(json_file_path, output_file_path):
    """
    从JSON文件中提取所有文本内容
    
    Args:
        json_file_path (str): 输入的JSON文件路径
        output_file_path (str): 输出的文本文件路径
        
    Returns:
        dict: 提取统计信息
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        extracted_texts = []
        stats = {
            'text': 0,
            'image': 0,
            'equation': 0,
            'table': 0,
            'total_items': len(data),
            'empty_items': 0
        }
        
        print(f"处理文件: {Path(json_file_path).name}")
        print(f"开始处理 {len(data)} 个内容项...")
        
        for i, item in enumerate(data):
            item_type = item.get('type', 'unknown')
            
            if item_type == 'text':
                # 提取普通文本
                text_content = clean_text(item.get('text', ''))
                if text_content:
                    extracted_texts.append(text_content)
                    stats['text'] += 1
                else:
                    stats['empty_items'] += 1
                    
            elif item_type == 'image':
                # 提取图片相关文本
                image_texts = []
                
                # 图片说明
                captions = item.get('image_caption', [])
                for caption in captions:
                    cleaned_caption = clean_text(caption)
                    if cleaned_caption:
                        image_texts.append(cleaned_caption)
                
                # 图片脚注
                footnotes = item.get('image_footnote', [])
                for footnote in footnotes:
                    cleaned_footnote = clean_text(footnote)
                    if cleaned_footnote:
                        image_texts.append(cleaned_footnote)
                
                if image_texts:
                    extracted_texts.extend(image_texts)
                    stats['image'] += 1
                else:
                    stats['empty_items'] += 1
                    
            elif item_type == 'equation':
                # 提取公式文本
                equation_text = clean_text(item.get('text', ''))
                
                if equation_text:
                    extracted_texts.append(equation_text)
                    stats['equation'] += 1
                else:
                    stats['empty_items'] += 1
                    
            elif item_type == 'table':
                # 提取表格相关文本
                table_texts = []
                
                # 表格标题
                captions = item.get('table_caption', [])
                for caption in captions:
                    cleaned_caption = clean_text(caption)
                    if cleaned_caption:
                        table_texts.append(cleaned_caption)
                
                # 表格内容
                table_body = item.get('table_body', '')
                if table_body:
                    clean_body = clean_html_table(table_body)
                    if clean_body:
                        table_texts.append(clean_body)
                
                # 表格脚注
                footnotes = item.get('table_footnote', [])
                for footnote in footnotes:
                    cleaned_footnote = clean_text(footnote)
                    if cleaned_footnote:
                        table_texts.append(cleaned_footnote)
                
                if table_texts:
                    extracted_texts.extend(table_texts)
                    stats['table'] += 1
                else:
                    stats['empty_items'] += 1
        
        # 过滤掉空文本并写入输出文件
        non_empty_texts = [text for text in extracted_texts if text.strip()]
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # 写入提取的内容，每个文本块之间用单个换行分隔
            for i, text in enumerate(non_empty_texts):
                f.write(text)
                # 如果不是最后一个文本块，添加换行
                if i < len(non_empty_texts) - 1:
                    f.write('\n')
        
        # 更新统计信息
        stats['final_text_blocks'] = len(non_empty_texts)
        
        print(f"提取完成！结果已保存到: {output_file_path}")
        return stats
            
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误：JSON文件格式不正确 - {json_file_path}")
        return None
    except Exception as e:
        print(f"错误：处理文件 {json_file_path} 时出现异常 - {str(e)}")
        return None


def process_multiple_files(json_files):
    """
    批量处理多个JSON文件
    
    Args:
        json_files (list): JSON文件路径列表
    """
    script_dir = Path(__file__).parent
    all_stats = {}
    successful_files = 0
    
    print("批量论文文本提取工具 (优化token版本)")
    print("=" * 60)
    
    for json_file in json_files:
        json_path = script_dir / json_file
        
        if not json_path.exists():
            print(f"警告：文件不存在 - {json_file}")
            continue
        
        # 生成输出文件名
        base_name = json_path.stem  # 获取不带扩展名的文件名
        if base_name.endswith('_content_list'):
            output_name = base_name.replace('_content_list', '_extracted_text.txt')
        else:
            output_name = f"{base_name}_extracted_text.txt"
        
        output_path = script_dir / output_name
        
        print(f"\n处理: {json_file} -> {output_name}")
        print("-" * 60)
        
        # 执行提取
        stats = extract_text_from_json(str(json_path), str(output_path))
        
        if stats:
            all_stats[json_file] = stats
            successful_files += 1
            print("提取统计:")
            print(f"  总项目: {stats['total_items']}")
            print(f"  文本项目: {stats['text']}")
            print(f"  图片项目: {stats['image']}")
            print(f"  公式项目: {stats['equation']}")
            print(f"  表格项目: {stats['table']}")
            print(f"  空项目: {stats['empty_items']}")
            print(f"  最终文本块: {stats['final_text_blocks']}")
        else:
            print("处理失败！")
    
    # 输出总结
    print("\n" + "=" * 60)
    print("批量处理完成总结:")
    print(f"成功处理文件数: {successful_files}/{len(json_files)}")
    
    if all_stats:
        print("\n各文件处理统计:")
        for filename, stats in all_stats.items():
            print(f"\n{filename}:")
            print(f"  总项目: {stats['total_items']}, 有效文本块: {stats['final_text_blocks']}")
            print(f"  文本: {stats['text']}, 图片: {stats['image']}, "
                  f"公式: {stats['equation']}, 表格: {stats['table']}, 空项目: {stats['empty_items']}")


def main():
    """主函数"""
    script_dir = Path(__file__).parent
    
    # 预定义的JSON文件列表
    target_files = [
        "paper_test_1_content_list.json",
        "paper_test_2_content_list.json"
    ]
    
    # 检查哪些文件存在
    existing_files = []
    for filename in target_files:
        if (script_dir / filename).exists():
            existing_files.append(filename)
    
    if not existing_files:
        print("错误：找不到任何目标JSON文件")
        print("请确保以下文件存在于当前目录:")
        for filename in target_files:
            print(f"  - {filename}")
        return
    
    if len(existing_files) < len(target_files):
        print("警告：部分文件不存在")
        print("找到的文件:")
        for filename in existing_files:
            print(f"  ✓ {filename}")
        print("缺失的文件:")
        for filename in target_files:
            if filename not in existing_files:
                print(f"  ✗ {filename}")
        print()
    
    # 处理找到的文件
    process_multiple_files(existing_files)


if __name__ == "__main__":
    main() 