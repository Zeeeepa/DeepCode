#!/usr/bin/env python3
"""
Sound generation script for Snake game.
This script generates simple sound effects using pygame's sound synthesis.
"""

import pygame
import numpy as np
import os

def generate_eat_sound():
    """Generate a pleasant eating sound effect."""
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    
    # Parameters for eat sound
    duration = 0.2  # seconds
    sample_rate = 22050
    
    # Generate a pleasant "pop" sound with multiple harmonics
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Base frequency and harmonics
    freq1 = 800  # Main frequency
    freq2 = 1200  # Higher harmonic
    freq3 = 600   # Lower harmonic
    
    # Create waveform with envelope
    envelope = np.exp(-t * 8)  # Exponential decay
    wave1 = np.sin(2 * np.pi * freq1 * t) * envelope * 0.4
    wave2 = np.sin(2 * np.pi * freq2 * t) * envelope * 0.3
    wave3 = np.sin(2 * np.pi * freq3 * t) * envelope * 0.3
    
    # Combine waves
    wave = wave1 + wave2 + wave3
    
    # Normalize and convert to 16-bit
    wave = np.clip(wave * 32767, -32767, 32767).astype(np.int16)
    
    # Create stereo sound - ensure C-contiguous array
    stereo_wave = np.column_stack((wave, wave))
    stereo_wave = np.ascontiguousarray(stereo_wave, dtype=np.int16)
    
    # Create pygame sound
    sound = pygame.sndarray.make_sound(stereo_wave)
    return sound

def generate_game_over_sound():
    """Generate a game over sound effect."""
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    
    # Parameters for game over sound
    duration = 1.0  # seconds
    sample_rate = 22050
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a descending tone sequence
    frequencies = [440, 370, 311, 262]  # Descending notes
    wave = np.zeros_like(t)
    
    segment_length = len(t) // len(frequencies)
    
    for i, freq in enumerate(frequencies):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, len(t))
        segment_t = t[start_idx:end_idx]
        
        # Create envelope for this segment
        envelope = np.exp(-segment_t * 3) * (1 - segment_t / duration)
        segment_wave = np.sin(2 * np.pi * freq * segment_t) * envelope
        
        wave[start_idx:end_idx] = segment_wave
    
    # Add some noise for dramatic effect
    noise = np.random.normal(0, 0.05, len(wave))
    wave = wave + noise
    
    # Normalize and convert to 16-bit
    wave = np.clip(wave * 32767, -32767, 32767).astype(np.int16)
    
    # Create stereo sound - ensure C-contiguous array
    stereo_wave = np.column_stack((wave, wave))
    stereo_wave = np.ascontiguousarray(stereo_wave, dtype=np.int16)
    
    # Create pygame sound
    sound = pygame.sndarray.make_sound(stereo_wave)
    return sound

def save_sounds():
    """Generate and save sound files."""
    try:
        # Initialize pygame mixer
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Generate eat sound
        print("Generating eat sound...")
        eat_sound = generate_eat_sound()
        eat_path = os.path.join(script_dir, "eat.wav")
        pygame.mixer.Sound.save(eat_sound, eat_path)
        print(f"Saved eat sound to: {eat_path}")
        
        # Generate game over sound
        print("Generating game over sound...")
        game_over_sound = generate_game_over_sound()
        game_over_path = os.path.join(script_dir, "game_over.wav")
        pygame.mixer.Sound.save(game_over_sound, game_over_path)
        print(f"Saved game over sound to: {game_over_path}")
        
        print("Sound generation completed successfully!")
        
    except Exception as e:
        print(f"Error generating sounds: {e}")
        print("This is normal if pygame or numpy are not installed.")
        print("The game will work without sound effects.")

if __name__ == "__main__":
    save_sounds()