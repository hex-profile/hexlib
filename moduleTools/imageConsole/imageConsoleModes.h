#pragma once

//================================================================
//
// DisplayMode
//
//================================================================

enum class DisplayMode
{
    // Display an image upsampled to the original frame size.
    Fullscreen, 

    // Display a true-sized image at the screen center,
    // on top of black image of the original frame size.
    Centered, 

    // Display a true-sized image at the left upper corner.
    Original, 
    
    COUNT
};

//================================================================
//
// VectorMode
//
//================================================================

enum class VectorMode
{
    // Visualize vector magnitude as brightness and vector angle as color.
    Color,

    // Visualize vector magnitude as grayscale brightness.
    Magnitude,

    // Visualize only X or Y vector part as a signed grayscale image.
    OnlyX,
    OnlyY,

    COUNT
};
