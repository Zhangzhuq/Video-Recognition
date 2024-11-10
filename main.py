from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
import os
import cv2
import ollama
import time
from pathlib import Path
import asyncio
import json
import subprocess
import torch
import numpy as np

# 全局常量
MODEL_NAME = 'llama3.2-vision'
UPLOAD_DIR = Path("uploads")
FRAMES_DIR = Path("frames")
MODELS_DIR = Path("models--meta-llama--Llama-3.2-11B-Vision")

def check_gpu():
    """检查 GPU 状态"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # 转换为GB
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_free = memory_total - memory_allocated
        return True, {
            "name": gpu_name,
            "memory_total": f"{memory_total:.2f}GB",
            "memory_free": f"{memory_free:.2f}GB"
        }
    return False, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    print("\n" + "="*50)
    print("正在初始化 VideoRecognition 应用...")
    print("="*50 + "\n")
    
    # 创建必要的目录
    UPLOAD_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)
    
    # 1. 检查 GPU 状态
    print("1. 检查 GPU 状态...")
    gpu_available, gpu_info = check_gpu()
    if gpu_available:
        print("√ 检测到可用 GPU:")
        print(f"  - GPU型号: {gpu_info['name']}")
        print(f"  - 总显存: {gpu_info['memory_total']}")
        print(f"  - 可用显存: {gpu_info['memory_free']}")
    else:
        print("× 未检测到可用 GPU，将使用 CPU 模式运行")
        print("警告: CPU 模式下性能可能较差")
    
    # 2. 检查并启动 Ollama 服务
    print("\n2. 检查 Ollama 服务状态...")
    try:
        subprocess.run(['ollama', 'list'], check=True, capture_output=True)
        print("√ Ollama 服务已在运行")
    except subprocess.CalledProcessError:
        print("× Ollama 服务未运行")
        print("正在启动 Ollama 服务...")
        try:
            if gpu_available:
                os.environ['OLLAMA_CUDA'] = '1'  # 启用 CUDA 支持
                print("已启用 CUDA 支持")
            subprocess.Popen(['ollama', 'serve'])
            print("等待服务启动中...")
            time.sleep(5)
            subprocess.run(['ollama', 'list'], check=True, capture_output=True)
            print("√ Ollama 服务启动成功")
        except Exception as e:
            print("× Ollama 服务启动失败")
            print(f"错误信息: {e}")
            print("请确保已正确安装 Ollama")
            raise e
    
    # 3. 检查模型状态
    print("\n3. 检查 llama3.2-vision 模型状态...")
    try:
        models = ollama.list()
        available_models = [model['name'] for model in models.get('models', [])]
        
        if MODEL_NAME in available_models:
            print(f"√ 模型 {MODEL_NAME} 已存在")
        else:
            print(f"× 未找到模型 {MODEL_NAME}")
            print("正在下载模型，这可能需要几分钟时间...")
            try:
                ollama.pull(MODEL_NAME)
                print(f"√ 模型 {MODEL_NAME} 下载成功")
            except Exception as e:
                print(f"× 模型下载失败: {e}")
                print("请确保网络连接正常，并手动运行:")
                print(f"ollama pull {MODEL_NAME}")
                raise e
    except Exception as e:
        print(f"× 模型检查失败: {e}")
        raise e
    
    print("\n4. 初始化完成，服务准备就绪！")
    print("="*50)
    mode = "GPU 加速模式" if gpu_available else "CPU 模式"
    print(f"VideoRecognition 运行于 {mode}")
    print(f"请访问 http://localhost:8000 开始使用")
    print("="*50 + "\n")
    
    yield
    
    print("\n正在关闭应用...")

# 创建 FastAPI 应用实例
app = FastAPI(lifespan=lifespan)

# 设置模板和静态文件
templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

def apply_clahe(image):
    # 检查图像是否为彩色图像
    if len(image.shape) == 3:
        # 将彩色图像转换为LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 分离通道
        l, a, b = cv2.split(lab)
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # 只对L通道进行CLAHE处理
        l = clahe.apply(l)
        
        # 合并通道
        lab = cv2.merge((l,a,b))
        
        # 转换回BGR色彩空间
        enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_image
    else:
        # 如果是灰度图像，直接应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

async def analyze_image(image_path: str, object_str: str):
    """异步版本的图像分析函数"""
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
        # 配置 Ollama 选项以最大化 GPU 利用率
        options = {
            'numa': True if torch.cuda.is_available() else False,
            'num_gpu': 1 if torch.cuda.is_available() else 0,
            'num_thread': 8,  # 增加线程数
            'num_ctx': 4096,  # 增加上下文长度
            'num_batch': 512,  # 增加批处理大小
            'num_keep': 48,   # 保持的上下文数量
            'rope_frequency_base': 1e6,  # RoPE 频率基数
            'rope_frequency_scale': 1.0, # RoPE 频率缩放
            'mmap': True      # 启用内存映射
        }
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': prompt_str,
                'images': [image_path]
            }],
            options=options
        )

        response_text = response['message']['content']
        response_lines = response_text.strip().split('\n')

        answer = None
        description = None
        confidence = 10

        for line in response_lines:
            line = line.strip()
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip().upper()
            elif any(line.lower().startswith(prefix) for prefix in
                     ['description:', 'reasoning:', 'alternative description:']):
                description = line.split(':', 1)[1].strip()
            elif line.lower().startswith('confidence:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 10

        return answer == "YES" and confidence >= 7, description, confidence
    except Exception as e:
        print(f"图像分析错误: {str(e)}")
        return False, f"分析出错: {str(e)}", 0

def preprocess_image(image_path):
    """图像预处理函数"""
    img = cv2.imread(image_path)
    if img is None:
        return False

    # 使用 GPU 加速预处理
    final = apply_clahe(img)
    cv2.imwrite(image_path, final, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return True

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_video(
        video: UploadFile = File(...),
        object_str: str = Form(...)
):
    try:
        # 保存上传的视频
        video_path = UPLOAD_DIR / video.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 为当前任务创建专门的帧目录
        task_frames_dir = FRAMES_DIR / video.filename.split('.')[0]
        task_frames_dir.mkdir(exist_ok=True)

        # 异步生成分析结果
        async def generate_results():
            cap = cv2.VideoCapture(str(video_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0

            try:
                while True:
                    success, frame = cap.read()
                    if not success:
                        break

                    if frame_count % fps == 0:  # 每秒处理一帧
                        current_second = frame_count // fps
                        frame_path = str(task_frames_dir / f"frame_{current_second}.jpg")
                        cv2.imwrite(frame_path, frame)

                        if preprocess_image(frame_path):
                            is_match, description, confidence = await analyze_image(frame_path, object_str)

                            result = {
                                "status": "success",
                                "frame": {
                                    "second": current_second,
                                    "is_match": is_match,
                                    "description": description,
                                    "confidence": confidence,
                                    "frame_path": f"/frames/{video.filename.split('.')[0]}/frame_{current_second}.jpg"
                                }
                            }

                            yield json.dumps(result) + "\n"

                    frame_count += 1

            finally:
                cap.release()

        return StreamingResponse(generate_results(), media_type="application/json")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)