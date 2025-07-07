import re

def add_fontproperties_to_line(line):
    # 判断是否已包含 fontproperties
    if 'fontproperties' in line:
        return line

    # 匹配需要处理的函数
    patterns = [
        r'\.set_title\((.*?)\)',
        r'\.set_xlabel\((.*?)\)',
        r'\.set_ylabel\((.*?)\)',
        r'\.set_xticklabels\((.*?)\)',
        r'\.set_yticklabels\((.*?)\)',
        r'\.legend\((.*?)\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            # 获取参数内容
            args = match.group(1).strip()
            # 如果已经为空参数则特殊处理
            new_args = args + ', fontproperties=font_prop' if args else 'fontproperties=font_prop'
            new_line = re.sub(pattern, f".{pattern.split('(')[0][1:]}({new_args})", line)
            return new_line

    return line

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = [add_fontproperties_to_line(line) for line in lines]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"✅ 修改完成，输出文件: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 修改为你的原始文件名
    input_file = "app.py"
    output_file = "app_with_font.py"
    process_file(input_file, output_file)
