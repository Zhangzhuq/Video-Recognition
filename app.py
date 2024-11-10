import cv2
import os
import ollama
import time


def analyze_image(image_path, object_str):
    """
    分析单张图像，检测是否存在目标对象
    Args:
        image_path: 图像文件路径
        object_str: 要检测的目标对象描述

    Returns:
        tuple: (是否匹配, 描述文本, 置信度)
    """
    prompt_str = f"""Please analyze the image and answer the following questions:

1. Is there a {object_str} in the image?
2. If yes, describe its appearance and location in the image in detail.
3. If no, describe what you see in the image instead.
4. On a scale of 1-10, how confident are you in your answer?

Please structure your response as follows:
Answer: [YES/NO]
Description: [Your detailed description]
Confidence: [1-10]"""

    try:
        # 调用llama模型分析图像
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[
                {
                    'role': 'user',
                    'content': prompt_str,
                    'images': [image_path]
                }
            ]
        )

        print(f"等待模型分析中...")
        time.sleep(1)  # 短暂延迟确保响应完整

        # 获取并打印原始响应
        response_text = response['message']['content']
        print(f"Raw response: {response_text}")

        # 处理响应文本，移除Markdown格式符号
        response_text = response_text.replace('**', '')
        response_lines = response_text.strip().split('\n')

        # 从响应中提取关键信息
        answer = None
        description = None
        confidence = 10  # 默认置信度为10，因为模型没有明确返回置信度

        # 逐行解析响应内容
        for line in response_lines:
            line = line.strip()
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip().upper()
            # 同时匹配Description、Reasoning和Alternative Description
            elif any(line.lower().startswith(prefix) for prefix in
                     ['description:', 'reasoning:', 'alternative description:']):
                description = line.split(':', 1)[1].strip()
            elif line.lower().startswith('confidence:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 10  # 如果无法解析置信度，使用默认值

        # 检查是否获取到必要的信息
        if answer is None or description is None:
            raise ValueError("Response format is incomplete")

        print(f"解析结果 - 答案: {answer}, 描述: {description}, 置信度: {confidence}")

        # 返回分析结果
        return answer == "YES" and confidence >= 7, description, confidence
    except Exception as e:
        print(f"Error during image analysis: {e}")
        import traceback
        print(traceback.format_exc())
        return False, "Error occurred", 0


def preprocess_image(image_path):
    """
    图像预处理函数，增强图像质量
    Args:
        image_path: 图像文件路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # 转换颜色空间并进行对比度增强
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 保存处理后的图像
    cv2.imwrite(image_path, final, [cv2.IMWRITE_JPEG_QUALITY, 100])


def extract_and_analyze_frames(video_path, output_folder, object_str):
    """
    从视频中提取帧并分析是否包含目标对象
    Args:
        video_path: 视频文件路径
        output_folder: 帧图像保存文件夹
        object_str: 要检测的目标对象描述

    Returns:
        int or None: 找到目标的时间点（秒），未找到返回None
    """
    # 创建输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None

    # 获取视频FPS
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    consecutive_matches = 0
    match_threshold = 1  # 连续匹配阈值
    cool_down_time = 2  # 每帧分析后的冷却时间（秒）

    print(f"开始分析视频，FPS: {fps}")

    try:
        while True:
            # 读取视频帧
            success, frame = video.read()
            if not success:
                break

            # 每秒处理一帧
            if frame_count % fps == 0:
                print(f"\n处理第 {frame_count // fps} 秒的帧")

                # 保存当前帧
                output_filename = os.path.join(output_folder, f"frame_{frame_count // fps}.jpg")
                output_filename = os.path.abspath(output_filename)

                cv2.imwrite(output_filename, frame)
                print(f"已保存帧到: {output_filename}")

                # 预处理图像
                preprocess_image(output_filename)
                print("已完成图像预处理")

                print("开始分析图像...")
                print(f"使用图像路径: {output_filename}")

                # 检查文件是否存在
                if not os.path.exists(output_filename):
                    print(f"警告: 文件不存在: {output_filename}")
                    continue

                # 分析图像
                is_match, description, confidence = analyze_image(output_filename, object_str)
                print(f"分析完成 - 匹配: {is_match}, 置信度: {confidence}")
                print(f"描述: {description}")

                # 处理匹配结果
                if is_match:
                    consecutive_matches += 1
                    print(f"潜在匹配 - 时间: 第 {frame_count // fps} 秒")
                    print(f"描述: {description}")
                    print(f"置信度: {confidence}")

                    # 如果连续匹配次数达到阈值，返回结果并退出
                    if consecutive_matches >= match_threshold:
                        match_time = frame_count // fps - match_threshold + 1
                        print(f"找到连续匹配！时间: 第 {match_time} 秒到第 {frame_count // fps} 秒")
                        video.release()  # 释放视频资源
                        return match_time  # 直接返回结果
                else:
                    consecutive_matches = 0

                # 分析完一帧后的冷却时间
                print(f"等待 {cool_down_time} 秒进行显卡冷却...")
                time.sleep(cool_down_time)

            frame_count += 1

    finally:
        # 确保视频资源被释放
        video.release()

    print(f"未找到匹配的图像。共分析了 {frame_count // fps} 张图像。")
    return None


# 主程序入口
if __name__ == "__main__":
    # 设置参数
    video_path = "./a.mp4"
    output_folder = "output_frames"
    object_to_find = "A man riding a bicycle"

    print("开始运行视频分析程序...")
    # 运行分析
    result = extract_and_analyze_frames(video_path, output_folder, object_to_find)

    # 输出结果
    if result is not None:
        print(f"目标对象在视频的第 {result} 秒被找到。")
    else:
        print("在整个视频中未找到目标对象。")