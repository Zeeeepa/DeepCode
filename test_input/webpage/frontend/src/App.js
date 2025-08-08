import React, { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Container, Paper, Typography } from '@material-ui/core';
import ImageUploader from './components/ImageUploader';
import ImageEditor from './components/ImageEditor';
import ColorPicker from './components/ColorPicker';

const useStyles = makeStyles((theme) => ({
  root: {
    marginTop: theme.spacing(4),
    marginBottom: theme.spacing(4),
  },
  paper: {
    padding: theme.spacing(3),
  },
  title: {
    marginBottom: theme.spacing(3),
  },
}));

function App() {
  const classes = useStyles();
  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [backgroundColor, setBackgroundColor] = useState('#FFFFFF');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = (file) => {
    setSelectedImage(file);
    setProcessedImage(null);
    setError(null);
  };

  const handleBackgroundColorChange = (color) => {
    setBackgroundColor(color.hex);
    if (selectedImage) {
      processImage(selectedImage, color.hex);
    }
  };

  const processImage = async (image, bgColor) => {
    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', image);
    formData.append('background_color', bgColor.replace('#', ''));

    try {
      const response = await fetch('http://localhost:8000/api/remove-background', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Image processing failed');
      }

      const blob = await response.blob();
      setProcessedImage(URL.createObjectURL(blob));
    } catch (err) {
      setError('Failed to process image. Please try again.');
      console.error('Error processing image:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Container maxWidth="md" className={classes.root}>
      <Paper className={classes.paper} elevation={3}>
        <Typography variant="h4" component="h1" className={classes.title}>
          ID Photo Background Tool
        </Typography>
        
        <ImageUploader 
          onImageSelect={handleImageUpload}
          isProcessing={isProcessing}
        />

        <ColorPicker
          color={backgroundColor}
          onChange={handleBackgroundColorChange}
        />

        {error && (
          <Typography color="error" style={{ marginTop: 16 }}>
            {error}
          </Typography>
        )}

        {(selectedImage || processedImage) && (
          <ImageEditor
            originalImage={selectedImage ? URL.createObjectURL(selectedImage) : null}
            processedImage={processedImage}
            isProcessing={isProcessing}
          />
        )}
      </Paper>
    </Container>
  );
}

export default App;