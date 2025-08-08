import React, { useState } from 'react';
import PropTypes from 'prop-types';
import './ImageEditor.css';

const ImageEditor = ({
  onRemoveBackground,
  onReplaceBackground,
  isProcessing,
  originalImage,
  processedImage
}) => {
  // 确保backgroundColor始终有明确的初始值，避免undefined状态
  const [backgroundColor, setBackgroundColor] = useState('#ffffff');
  const [backgroundImage, setBackgroundImage] = useState(null);
  const [activeTab, setActiveTab] = useState('color'); // 'color' or 'image'
  const [fileInputKey, setFileInputKey] = useState(0); // 用于重置file input

  // 添加空值检查，确保受控组件的稳定性
  const handleColorChange = (e) => {
    const value = e.target.value;
    if (value !== undefined && value !== null) {
      setBackgroundColor(value);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setBackgroundImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleBackgroundReplace = () => {
    if (activeTab === 'color') {
      onReplaceBackground(backgroundColor, null);
    } else {
      onReplaceBackground(null, backgroundImage);
    }
  };

  // 清空背景图片并重置file input
  const clearBackgroundImage = () => {
    setBackgroundImage(null);
    setFileInputKey(prev => prev + 1); // 通过改变key来重置file input
  };

  return (
    <div className="image-editor">
      <div className="editor-controls">
        <button
          onClick={onRemoveBackground}
          disabled={isProcessing || !originalImage}
          className="editor-button"
        >
          Remove Background
        </button>

        <div className="background-options">
          <div className="tab-buttons">
            <button
              className={`tab-button ${activeTab === 'color' ? 'active' : ''}`}
              onClick={() => setActiveTab('color')}
            >
              Color
            </button>
            <button
              className={`tab-button ${activeTab === 'image' ? 'active' : ''}`}
              onClick={() => setActiveTab('image')}
            >
              Image
            </button>
          </div>

          {activeTab === 'color' ? (
            <div className="color-picker">
              <label htmlFor="background-color">Background Color:</label>
              <input
                type="color"
                id="background-color"
                value={backgroundColor}
                onChange={handleColorChange}
                disabled={isProcessing || !processedImage}
              />
            </div>
          ) : (
            <div className="image-upload">
              <label htmlFor="background-image">Background Image:</label>
              <input
                key={fileInputKey} // 使用key来控制file input的重置
                type="file"
                id="background-image"
                accept="image/*"
                onChange={handleImageUpload}
                disabled={isProcessing || !processedImage}
              />
              {backgroundImage && (
                <button
                  type="button"
                  onClick={clearBackgroundImage}
                  className="clear-button"
                  disabled={isProcessing}
                >
                  Clear Image
                </button>
              )}
            </div>
          )}

          <button
            onClick={handleBackgroundReplace}
            disabled={isProcessing || !processedImage || (activeTab === 'image' && !backgroundImage)}
            className="editor-button"
          >
            Replace Background
          </button>
        </div>
      </div>

      {isProcessing && (
        <div className="processing-overlay">
          <div className="spinner"></div>
          <p>Processing image...</p>
        </div>
      )}
    </div>
  );
};

ImageEditor.propTypes = {
  onRemoveBackground: PropTypes.func.isRequired,
  onReplaceBackground: PropTypes.func.isRequired,
  isProcessing: PropTypes.bool.isRequired,
  originalImage: PropTypes.string,
  processedImage: PropTypes.string
};

export default ImageEditor;