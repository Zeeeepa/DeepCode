import React from 'react';
import PropTypes from 'prop-types';

const Preview = ({ originalImage = null, processedImage = null }) => {
  const previewStyle = {
    display: 'flex',
    justifyContent: 'space-around',
    alignItems: 'center',
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    margin: '20px 0'
  };

  const imageContainerStyle = {
    textAlign: 'center',
    flex: 1,
    margin: '0 10px'
  };

  const imageStyle = {
    maxWidth: '100%',
    maxHeight: '400px',
    borderRadius: '4px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  };

  const labelStyle = {
    marginTop: '10px',
    fontSize: '14px',
    color: '#666'
  };

  return (
    <div className="preview-container" style={previewStyle}>
      <div style={imageContainerStyle}>
        <h3>Original Image</h3>
        {originalImage ? (
          <>
            <img
              src={typeof originalImage === 'string' ? originalImage : URL.createObjectURL(originalImage)}
              alt="Original"
              style={imageStyle}
            />
            <p style={labelStyle}>Original photo</p>
          </>
        ) : (
          <p>No image uploaded</p>
        )}
      </div>

      <div style={imageContainerStyle}>
        <h3>Processed Image</h3>
        {processedImage ? (
          <>
            <img
              src={processedImage}
              alt="Processed"
              style={imageStyle}
            />
            <p style={labelStyle}>Processed result</p>
          </>
        ) : (
          <p>No processed image yet</p>
        )}
      </div>
    </div>
  );
};

Preview.propTypes = {
  originalImage: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.instanceOf(File)
  ]),
  processedImage: PropTypes.string
};

export default Preview;