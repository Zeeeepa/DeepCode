import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { 
  Paper,
  Typography,
  Grid,
  IconButton,
  Tooltip
} from '@material-ui/core';
import { SketchPicker } from 'react-color';

const useStyles = makeStyles((theme) => ({
  root: {
    padding: theme.spacing(2),
    marginTop: theme.spacing(2),
  },
  colorPreview: {
    width: '30px',
    height: '30px',
    borderRadius: '50%',
    border: '2px solid #fff',
    boxShadow: '0 0 0 1px rgba(0,0,0,0.1)',
    cursor: 'pointer',
  },
  presetColor: {
    width: '25px',
    height: '25px',
    borderRadius: '50%',
    margin: theme.spacing(0.5),
    cursor: 'pointer',
    border: '2px solid #fff',
    boxShadow: '0 0 0 1px rgba(0,0,0,0.1)',
    '&:hover': {
      transform: 'scale(1.1)',
    },
  },
  popover: {
    position: 'absolute',
    zIndex: '2',
  },
  cover: {
    position: 'fixed',
    top: '0px',
    right: '0px',
    bottom: '0px',
    left: '0px',
  },
}));

const presetColors = [
  '#FFFFFF', // White
  '#87CEEB', // Sky Blue
  '#C0C0C0', // Silver
  '#FFB6C1', // Light Pink
  '#98FB98', // Pale Green
  '#DDA0DD', // Plum
  '#F0E68C', // Khaki
  '#E6E6FA', // Lavender
];

const ColorPicker = ({ color, onChange }) => {
  const classes = useStyles();
  const [displayColorPicker, setDisplayColorPicker] = React.useState(false);

  const handleClick = () => {
    setDisplayColorPicker(!displayColorPicker);
  };

  const handleClose = () => {
    setDisplayColorPicker(false);
  };

  const handleChange = (newColor) => {
    onChange(newColor.hex);
  };

  const handlePresetColorClick = (presetColor) => {
    onChange(presetColor);
  };

  return (
    <Paper className={classes.root}>
      <Typography variant="h6" gutterBottom>
        Background Color
      </Typography>
      
      <Grid container spacing={2} alignItems="center">
        <Grid item>
          <Tooltip title="Current Color">
            <div
              className={classes.colorPreview}
              style={{ backgroundColor: color }}
              onClick={handleClick}
            />
          </Tooltip>
        </Grid>
        
        <Grid item xs>
          <Typography variant="body2" color="textSecondary">
            Click to change color or select from presets below
          </Typography>
        </Grid>
      </Grid>

      <Grid container spacing={1} style={{ marginTop: '8px' }}>
        {presetColors.map((presetColor) => (
          <Grid item key={presetColor}>
            <Tooltip title={presetColor}>
              <div
                className={classes.presetColor}
                style={{ backgroundColor: presetColor }}
                onClick={() => handlePresetColorClick(presetColor)}
              />
            </Tooltip>
          </Grid>
        ))}
      </Grid>

      {displayColorPicker && (
        <div className={classes.popover}>
          <div className={classes.cover} onClick={handleClose} />
          <SketchPicker
            color={color}
            onChange={handleChange}
            disableAlpha
          />
        </div>
      )}
    </Paper>
  );
};

export default ColorPicker;