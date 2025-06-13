#!/usr/bin/env python3
"""
人脸检测测试脚本
用于诊断LatentSync项目中的人脸检测问题
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from latentsync.utils.face_detector import FaceDetector
from latentsync.utils.util import read_video


def test_single_image(image_path, output_dir="./test_output"):
    """测试单张图片的人脸检测"""
    print(f"Testing face detection on image: {image_path}")
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {image_rgb.shape}, dtype: {image_rgb.dtype}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试不同的配置
    configs = [
        {
            'name': 'default',
            'min_face_size': 50,
            'min_face_height': 80,
            'aspect_ratio_range': (0.2, 1.5),
            'debug': True
        },
        {
            'name': 'lenient',
            'min_face_size': 30,
            'min_face_height': 50,
            'aspect_ratio_range': (0.1, 3.0),
            'debug': True
        },
        {
            'name': 'very_lenient',
            'min_face_size': 20,
            'min_face_height': 30,
            'aspect_ratio_range': (0.05, 5.0),
            'debug': True
        }
    ]
    
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing configuration: {config['name']}")
        print(f"{'='*50}")
        
        try:
            detector = FaceDetector(device="cuda", **{k: v for k, v in config.items() if k != 'name'})
            
            for threshold in thresholds:
                print(f"\nTesting threshold: {threshold}")
                bbox, landmarks = detector(image_rgb, threshold=threshold)
                
                if bbox is not None:
                    print(f"✓ Face detected with config '{config['name']}' and threshold {threshold}")
                    
                    # 保存检测结果
                    result_image = image.copy()
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制关键点
                    if landmarks is not None:
                        for point in landmarks:
                            cv2.circle(result_image, tuple(point.astype(int)), 1, (0, 0, 255), -1)
                    
                    output_path = Path(output_dir) / f"result_{config['name']}_th{threshold}.jpg"
                    cv2.imwrite(str(output_path), result_image)
                    print(f"Result saved to: {output_path}")
                    
                    return True  # 成功检测到人脸
                else:
                    print(f"✗ No face detected with config '{config['name']}' and threshold {threshold}")
                    
        except Exception as e:
            print(f"Error with config '{config['name']}': {str(e)}")
    
    return False


def test_video(video_path, max_frames=10, output_dir="./test_output"):
    """测试视频的人脸检测"""
    print(f"Testing face detection on video: {video_path}")
    
    try:
        # 读取视频帧
        video_frames = read_video(video_path, change_fps=False)
        print(f"Video loaded: {len(video_frames)} frames")
        
        # 测试前几帧
        test_frames = min(max_frames, len(video_frames))
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用较宽松的配置
        detector = FaceDetector(
            device="cuda",
            min_face_size=30,
            min_face_height=50,
            aspect_ratio_range=(0.1, 3.0),
            detection_threshold=0.3,
            debug=True
        )
        
        success_count = 0
        
        for i in range(test_frames):
            print(f"\n--- Testing frame {i+1}/{test_frames} ---")
            frame = video_frames[i]
            
            # 保存原始帧
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(Path(output_dir) / f"frame_{i:03d}_original.jpg"), frame_bgr)
            
            # 测试人脸检测
            bbox, landmarks = detector(frame, threshold=0.3)
            
            if bbox is not None:
                success_count += 1
                print(f"✓ Face detected in frame {i+1}")
                
                # 保存检测结果
                result_frame = frame_bgr.copy()
                x1, y1, x2, y2 = bbox
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if landmarks is not None:
                    for point in landmarks:
                        cv2.circle(result_frame, tuple(point.astype(int)), 1, (0, 0, 255), -1)
                
                cv2.imwrite(str(Path(output_dir) / f"frame_{i:03d}_detected.jpg"), result_frame)
            else:
                print(f"✗ No face detected in frame {i+1}")
        
        print(f"\nSummary: {success_count}/{test_frames} frames had detectable faces")
        return success_count > 0
        
    except Exception as e:
        print(f"Error testing video: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test face detection for LatentSync")
    parser.add_argument("input_path", help="Path to image or video file")
    parser.add_argument("--output_dir", default="./test_output", help="Output directory for test results")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum frames to test for video")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    print("LatentSync Face Detection Test")
    print("=" * 50)
    
    # 检查文件类型
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    if input_path.suffix.lower() in image_extensions:
        success = test_single_image(str(input_path), args.output_dir)
    elif input_path.suffix.lower() in video_extensions:
        success = test_video(str(input_path), args.max_frames, args.output_dir)
    else:
        print(f"Error: Unsupported file type {input_path.suffix}")
        return
    
    if success:
        print("\n✓ Face detection test completed successfully!")
        print(f"Check results in: {args.output_dir}")
    else:
        print("\n✗ Face detection test failed!")
        print("Suggestions:")
        print("1. Check if the input contains clear, visible faces")
        print("2. Ensure good lighting and image quality")
        print("3. Try different input files")
        print("4. Check if CUDA and InsightFace models are properly installed")


if __name__ == "__main__":
    main() 