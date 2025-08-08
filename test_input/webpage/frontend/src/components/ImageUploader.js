import React, { useRef } from 'react';
import { 
  Button,
  Box,
  Typography,
  makeStyles
} from '@material-ui/core';
import CloudUploadIcon from '@material-ui/icons/CloudUpload';

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: theme.spacing(3),
    backgroundColor: '#f5f5f5',
    borderRadius: theme.spacing(1),
    border: '2px dashed #ccc',
    cursor: 'pointer',
    '&:hover': {
      backgroundColor: '#eee',
      borderColor: theme.palette.primary.main,
    },
  },
  input: {
    display: 'none',
  },
  icon: {
    fontSize: 48,
    marginBottom: theme.spacing(1),
    color: theme.palette.primary.main,
  },
  preview: {
    maxWidth: '100%',
    maxHeight: '300px',
    marginTop: theme.spacing(2),
  },
}));

const ImageUploader = ({ onImageUpload, selectedImage }) => {
  const classes = useStyles();
  const fileInputRef = useRef();

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleFile = (file) => {
    if (file.type.startsWith('image/')) {
      onImageUpload(file);
    } else {
      alert('Please upload an image file');
    }
  };

  return (
    <Box
      className={classes.root}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input
        type="file"
        className={classes.input}
        ref={fileInputRef}
        onChange={handleFileInput}
        accept="image/*"
      />
      
      {!selectedImage ? (
        <>
          <CloudUploadIcon className={classes.icon} />
          <Typography variant="h6" gutterBottom>
            Drag and drop an image here
          </Typography>
          <Typography variant="body2" color="textSecondary">
            or click to select a file
          </Typography>
          <Button
            variant="contained"
            color="primary"
            style={{ marginTop: '16px' }}
          >
            Upload Image
          </Button>
        </>
      ) : (
        <img
          src={URL.createObjectURL(selectedImage)}
          alt="Preview"
          className={classes.preview}
        />
      )}
    </Box>
  );
};

export default ImageUploader;