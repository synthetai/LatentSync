"""
人脸检测配置文件
根据你的具体需求调整这些参数
"""

# 人脸检测配置
FACE_DETECTION_CONFIGS = {
    # 默认配置（严格）
    'default': {
        'min_face_size': 50,              # 最小人脸宽度（像素）
        'min_face_height': 80,            # 最小人脸高度（像素）
        'aspect_ratio_range': (0.2, 1.5), # 宽高比范围
        'detection_threshold': 0.5,       # 检测置信度阈值
        'debug': False
    },
    
    # 宽松配置（推荐）
    'lenient': {
        'min_face_size': 30,              # 降低最小人脸尺寸要求
        'min_face_height': 50,            # 降低最小人脸高度要求
        'aspect_ratio_range': (0.1, 3.0), # 更宽松的宽高比范围
        'detection_threshold': 0.3,       # 降低检测阈值
        'debug': True                     # 启用调试信息
    },
    
    # 非常宽松配置（用于困难情况）
    'very_lenient': {
        'min_face_size': 20,              # 进一步降低最小人脸尺寸
        'min_face_height': 30,            # 进一步降低最小人脸高度
        'aspect_ratio_range': (0.05, 5.0), # 非常宽松的宽高比范围
        'detection_threshold': 0.2,       # 更低的检测阈值
        'debug': True
    },
    
    # 自定义配置（根据需要修改）
    'custom': {
        'min_face_size': 25,              # 根据你的视频调整
        'min_face_height': 40,            # 根据你的视频调整
        'aspect_ratio_range': (0.1, 2.5), # 根据你的视频调整
        'detection_threshold': 0.25,      # 根据你的视频调整
        'debug': True
    }
}

# 当前使用的配置（修改这里来切换配置）
CURRENT_CONFIG = 'lenient'  # 可选: 'default', 'lenient', 'very_lenient', 'custom'

def get_face_detection_config():
    """获取当前的人脸检测配置"""
    config = FACE_DETECTION_CONFIGS.get(CURRENT_CONFIG, FACE_DETECTION_CONFIGS['lenient'])
    print(f"Using face detection config: {CURRENT_CONFIG}")
    print(f"Config details: {config}")
    return config

# 使用说明
USAGE_INSTRUCTIONS = """
使用说明：

1. 如果遇到 "Face not detected" 错误，可以尝试以下步骤：
   - 将 CURRENT_CONFIG 改为 'lenient' 或 'very_lenient'
   - 根据你的视频特点调整 'custom' 配置中的参数
   
2. 参数说明：
   - min_face_size: 最小人脸宽度，数值越小越容易检测到小人脸
   - min_face_height: 最小人脸高度，数值越小越容易检测到小人脸
   - aspect_ratio_range: 宽高比范围，范围越宽越容易检测到各种形状的人脸
   - detection_threshold: 检测置信度阈值，数值越小越容易检测到人脸（但可能产生误检）
   - debug: 是否输出调试信息，建议设为 True 来诊断问题

3. 建议的调试流程：
   - 首先使用 'lenient' 配置
   - 如果还是失败，尝试 'very_lenient' 配置
   - 如果需要针对特定视频优化，修改 'custom' 配置并设置 CURRENT_CONFIG = 'custom'
   
4. 使用测试脚本：
   python test_face_detection.py your_video.mp4
   
   这会帮助你诊断具体的问题并找到最适合的配置。
"""

if __name__ == "__main__":
    print(USAGE_INSTRUCTIONS)
    print("\n当前配置:")
    print(get_face_detection_config()) 