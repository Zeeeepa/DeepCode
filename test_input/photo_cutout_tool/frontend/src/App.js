import React, { useState, useCallback } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import ImageUpload from './components/ImageUpload';
import ImageEditor from './components/ImageEditor';
import Preview from './components/Preview';
import axios from 'axios';

// 修改API基础URL，添加'/api'前缀以匹配后端路由配置
const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleImageUpload = async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      setIsProcessing(true);
      // 统一API端点路径格式，移除尾部斜杠以匹配后端路由定义
      const response = await axios.post(`${API_BASE_URL}/upload`, formData);
      
      setUploadedImage({
        filename: response.data.filename,
        preview: URL.createObjectURL(file)
      });
      toast.success('Image uploaded successfully!');
    } catch (error) {
      toast.error('Failed to upload image');
      console.error('Upload error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleBackgroundRemoval = async () => {
    if (!uploadedImage) return;

    try {
      setIsProcessing(true);
      // 统一API端点路径格式，移除尾部斜杠以匹配后端路由定义
      const response = await axios.post(
        `${API_BASE_URL}/remove-background/${uploadedImage.filename}`,
        {},
        { responseType: 'blob' }
      );
      
      setProcessedImage(URL.createObjectURL(response.data));
      toast.success('Background removed successfully!');
    } catch (error) {
      toast.error('Failed to remove background');
      console.error('Processing error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleBackgroundReplace = async (bgColor, bgImage) => {
    if (!uploadedImage) return;

    try {
      setIsProcessing(true);
      const formData = new FormData();
      if (bgColor) formData.append('bg_color', bgColor);
      if (bgImage) formData.append('bg_image', bgImage);

      // 统一API端点路径格式，移除尾部斜杠以匹配后端路由定义
      const response = await axios.post(
        `${API_BASE_URL}/replace-background/${uploadedImage.filename}`,
        formData,
        { responseType: 'blob' }
      );
      
      setProcessedImage(URL.createObjectURL(response.data));
      toast.success('Background replaced successfully!');
    } catch (error) {
      toast.error('Failed to replace background');
      console.error('Processing error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleOptimizeImage = async (maxSize) => {
    if (!uploadedImage) return;

    try {
      setIsProcessing(true);
      // 统一API端点路径格式，移除尾部斜杠以匹配后端路由定义
      const response = await axios.post(
        `${API_BASE_URL}/optimize/${uploadedImage.filename}?max_size=${maxSize}`,
        {},
        { responseType: 'blob' }
      );
      
      setProcessedImage(URL.createObjectURL(response.data));
      toast.success('Image optimized successfully!');
    } catch (error) {
      toast.error('Failed to optimize image');
      console.error('Optimization error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  // 使用useCallback优化cleanup函数，避免不必要的重新渲染
  const cleanup = useCallback(async () => {
    if (!uploadedImage) return;

    try {
      // 统一API端点路径格式，移除尾部斜杠以匹配后端路由定义
      await axios.delete(`${API_BASE_URL}/cleanup/${uploadedImage.filename}`);
      setUploadedImage(null);
      setProcessedImage(null);
    } catch (error) {
      console.error('Cleanup error:', error);
    }
  }, [uploadedImage]);

  // 修复useEffect Hook的依赖数组，添加cleanup和uploadedImage依赖项
  React.useEffect(() => {
    return () => {
      if (uploadedImage) {
        cleanup();
      }
    };
  }, [cleanup, uploadedImage]);

  return (
    <div className="app-container">
      <h1>ID Photo Background Manager</h1>
      
      <ToastContainer position="top-right" autoClose={3000} />
      
      <div className="main-content">
        {/* 修复：将prop名称从'onUpload'改为'onImageUpload'以匹配组件定义 */}
        <ImageUpload 
          onImageUpload={handleImageUpload}
          isProcessing={isProcessing}
        />
        
        {uploadedImage && (
          <ImageEditor
            onRemoveBackground={handleBackgroundRemoval}
            onReplaceBackground={handleBackgroundReplace}
            onOptimize={handleOptimizeImage}
            isProcessing={isProcessing}
            // 修复：添加必需的originalImage和processedImage props以避免controlled input警告
            originalImage={uploadedImage?.preview}
            processedImage={processedImage}
          />
        )}
        
        <Preview
          originalImage={uploadedImage?.preview}
          processedImage={processedImage}
        />
      </div>
    </div>
  );
}

export default App;