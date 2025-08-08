import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Paper, Typography, CircularProgress } from '@material-ui/core';
import { Alert } from '@material-ui/lab';

const useStyles = makeStyles((theme) => ({
  root: {
    padding: theme.spacing(2),
    marginTop: theme.spacing(2),
    marginBottom: theme.spacing(2),
  },
  imageContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: '300px',
    position: 'relative',
  },
  image: {
    maxWidth: '100%',
    maxHeight: '400px',
    objectFit: 'contain',
  },
  loading: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
  },
  previewContainer: {
    display: 'flex',
    gap: theme.spacing(2),
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  previewSection: {
    flex: '1 1 300px',
    maxWidth: '400px',
  },
}));

const ImageEditor = ({
  selectedImage,
  processedImage,
  backgroundColor,
  isProcessing,
  error,
}) => {
  const classes = useStyles();

  const renderImage = (image, title) => {
    if (!image) return null;

    return (
      <div className={classes.previewSection}>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <Paper className={classes.imageContainer}>
          <img
            src={typeof image === 'string' ? image : URL.createObjectURL(image)}
            alt={title}
            className={classes.image}
          />
        </Paper>
      </div>
    );
  };

  if (!selectedImage && !processedImage) {
    return (
      <Paper className={classes.root}>
        <Typography variant="body1" align="center">
          Please select an image to begin editing
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper className={classes.root}>
      {error && (
        <Alert severity="error" style={{ marginBottom: '16px' }}>
          {error}
        </Alert>
      )}
      
      <div className={classes.previewContainer}>
        {renderImage(selectedImage, 'Original Image')}
        
        <div className={classes.previewSection}>
          <Typography variant="h6" gutterBottom>
            Processed Image
          </Typography>
          <Paper 
            className={classes.imageContainer}
            style={{ backgroundColor: backgroundColor }}
          >
            {isProcessing ? (
              <CircularProgress className={classes.loading} />
            ) : (
              processedImage && (
                <img
                  src={processedImage}
                  alt="Processed"
                  className={classes.image}
                />
              )
            )}
          </Paper>
        </div>
      </div>
    </Paper>
  );
};

export default ImageEditor;