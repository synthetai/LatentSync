# LatentSync 人脸检测问题解决指南

如果你在运行LatentSync时遇到 "Face not detected" 错误，这个指南将帮助你诊断和解决问题。

## 🚨 常见错误信息

```
Error during processing: Face not detected
RuntimeError: Face not detected after 3 attempts with thresholds [0.3, 0.3, 0.2, 0.1].
```

## 🔧 已做的改进

我们对代码进行了以下改进来解决人脸检测问题：

### 1. 更宽松的检测参数
- 降低了最小人脸尺寸要求 (50→30像素)
- 降低了最小人脸高度要求 (80→50像素)
- 放宽了宽高比范围 (0.2-1.5 → 0.1-3.0)
- 降低了检测置信度阈值 (0.5→0.3)

### 2. 多重尝试机制
- 自动使用不同阈值进行多次检测尝试
- 渐进式降低检测要求
- 详细的调试信息输出

### 3. 更好的错误处理
- 提供详细的错误诊断信息
- 用户友好的错误提示
- 具体的解决建议

## 🛠️ 使用方法

### 方法1: 使用测试脚本诊断问题

```bash
# 测试单个视频文件
python test_face_detection.py your_video.mp4

# 测试图片文件
python test_face_detection.py your_image.jpg

# 指定输出目录
python test_face_detection.py your_video.mp4 --output_dir ./debug_output
```

测试脚本会：
- 测试不同的检测配置
- 保存检测结果图片
- 提供详细的诊断信息
- 给出具体的建议

### 方法2: 调整检测参数

编辑 `face_detection_config.py` 文件：

```python
# 修改当前使用的配置
CURRENT_CONFIG = 'very_lenient'  # 可选: 'default', 'lenient', 'very_lenient', 'custom'

# 或者自定义参数
FACE_DETECTION_CONFIGS['custom'] = {
    'min_face_size': 20,              # 更小的人脸尺寸要求
    'min_face_height': 30,            # 更小的人脸高度要求
    'aspect_ratio_range': (0.05, 5.0), # 更宽松的宽高比
    'detection_threshold': 0.1,       # 更低的置信度阈值
    'debug': True
}
```

### 方法3: 直接运行改进后的主程序

```bash
# 运行Gradio界面
python gradio_app.py

# 或者命令行推理
./inference.sh
```

## 📋 故障排除步骤

### 1. 检查视频质量
确保你的视频满足以下条件：
- ✅ 包含清晰可见的人脸
- ✅ 人脸没有被严重遮挡
- ✅ 光照条件良好（不要太暗或太亮）
- ✅ 人脸角度不要太侧或太仰/俯
- ✅ 人脸在画面中占据足够大的区域

### 2. 使用测试脚本
```bash
python test_face_detection.py your_video.mp4
```

查看输出信息，确认：
- InsightFace是否检测到人脸
- 哪些人脸被过滤掉了
- 过滤的原因是什么

### 3. 逐步调整参数
如果测试脚本显示人脸被过滤，按以下顺序调整：

1. **降低尺寸要求**：
   ```python
   'min_face_size': 20,
   'min_face_height': 30,
   ```

2. **放宽宽高比范围**：
   ```python
   'aspect_ratio_range': (0.05, 5.0),
   ```

3. **降低检测阈值**：
   ```python
   'detection_threshold': 0.1,
   ```

### 4. 预处理视频
如果以上方法都不行，考虑预处理视频：

```python
# 调整亮度和对比度
import cv2
cap = cv2.VideoCapture('input.mp4')
# ... 处理每帧，调整亮度对比度
```

## 🔍 调试信息解读

运行时会看到类似以下的调试信息：

```
Processing frame of size: 1920x1080
InsightFace detected 2 face(s)
Face 0: bbox=[100, 150, 300, 400], size=200x250, aspect_ratio=0.80, det_score=0.854
Face 1: bbox=[500, 200, 600, 350], size=100x150, aspect_ratio=0.67, det_score=0.234
Face 1 filtered out: confidence 0.234 < 0.3
After filtering: 1 face(s) remain
Selected face bbox: (95, 145, 305, 415)
```

从这个信息可以看出：
- 检测到了2个人脸
- Face 1因为置信度太低被过滤
- 最终选择了Face 0

## 📝 配置说明

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| min_face_size | 30 | 10-100 | 最小人脸宽度(像素) |
| min_face_height | 50 | 20-150 | 最小人脸高度(像素) |
| aspect_ratio_range | (0.1, 3.0) | (0.05, 5.0) | 宽高比范围 |
| detection_threshold | 0.3 | 0.1-0.8 | 检测置信度阈值 |

## 🎯 最佳实践

1. **先用demo视频测试**：确保环境配置正确
2. **使用测试脚本**：诊断具体问题
3. **逐步调整参数**：从宽松配置开始
4. **检查视频质量**：确保输入满足基本要求
5. **保存工作配置**：找到合适配置后保存

## ❓ 常见问题

**Q: 为什么demo视频可以但我的视频不行？**
A: 可能是视频质量、人脸大小或角度问题。使用测试脚本对比分析。

**Q: 调试信息显示检测到人脸但还是报错？**
A: 检查是否所有检测到的人脸都被过滤掉了，调整过滤参数。

**Q: 可以跳过人脸检测吗？**
A: 不可以，人脸检测是口型同步的必要步骤。

**Q: 如何提高检测成功率？**
A: 使用高质量视频，确保良好光照，人脸尽量正面且清晰。

## 📞 获取帮助

如果以上方法都无法解决问题，请：
1. 运行测试脚本并保存输出日志
2. 检查视频的基本信息（分辨率、帧率等）
3. 尝试使用不同的视频文件
4. 确认CUDA和InsightFace安装正确

记住：人脸检测的成功很大程度上取决于输入视频的质量！ 