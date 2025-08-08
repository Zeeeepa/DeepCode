import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import PropTypes from 'prop-types';

const ImageUpload = ({ onImageUpload, isProcessing = false }) => {
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
      }
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('File size should be less than 10MB');
        return;
      }
      onImageUpload(file);
    }
  }, [onImageUpload]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    multiple: false,
    disabled: isProcessing
  });

  // CSS-in-JS styles using React style objects
  const styles = {
    imageUploadContainer: {
      width: '100%',
      maxWidth: '600px',
      margin: '0 auto'
    },
    dropzone: {
      border: '2px dashed #cccccc',
      borderRadius: '4px',
      padding: '20px',
      textAlign: 'center',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      position: 'relative'
    },
    dropzoneActive: {
      borderColor: '#2196f3',
      backgroundColor: 'rgba(33, 150, 243, 0.1)'
    },
    dropzoneDisabled: {
      opacity: 0.7,
      cursor: 'not-allowed'
    },
    uploadContent: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '10px'
    },
    uploadIcon: {
      fontSize: '48px'
    },
    processingOverlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(255, 255, 255, 0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontWeight: 'bold'
    }
  };

  // Combine styles based on state
  const getDropzoneStyle = () => {
    let combinedStyle = { ...styles.dropzone };
    
    if (dragActive) {
      combinedStyle = { ...combinedStyle, ...styles.dropzoneActive };
    }
    
    if (isProcessing) {
      combinedStyle = { ...combinedStyle, ...styles.dropzoneDisabled };
    }
    
    return combinedStyle;
  };

  return (
    <div style={styles.imageUploadContainer}>
      <div
        {...getRootProps()}
        style={getDropzoneStyle()}
        onDragEnter={() => setDragActive(true)}
        onDragLeave={() => setDragActive(false)}
        onDrop={() => setDragActive(false)}
      >
        <input {...getInputProps()} />
        <div style={styles.uploadContent}>
          <i style={styles.uploadIcon}>📁</i>
          <p>Drag and drop an image here, or click to select</p>
          <em>Supports JPG, PNG and GIF (max 10MB)</em>
          {isProcessing && (
            <div style={styles.processingOverlay}>
              Processing...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

ImageUpload.propTypes = {
  onImageUpload: PropTypes.func.isRequired,
  isProcessing: PropTypes.bool
};

export default ImageUpload;