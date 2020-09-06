////////////////////////////////////////////////////////////
//
// SFML - Simple and Fast Multimedia Library
// Copyright (C) 2007-2009 Laurent Gomila (laurent.gom@gmail.com)
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it freely,
// subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented;
//    you must not claim that you wrote the original software.
//    If you use this software in a product, an acknowledgment
//    in the product documentation would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such,
//    and must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.
//
////////////////////////////////////////////////////////////

#ifndef SFML_IMAGE_HPP
#define SFML_IMAGE_HPP

////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <SFML/System/Resource.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <string>
#include <vector>


namespace sf
{
class Renderer;
class RenderImage;
class RenderWindow;

////////////////////////////////////////////////////////////
/// Image is the low-level class for loading and
/// manipulating images
////////////////////////////////////////////////////////////
class SFML_API Image : public Resource<Image>
{
public :

    ////////////////////////////////////////////////////////////
    /// Default constructor
    ///
    ////////////////////////////////////////////////////////////
    Image();

    ////////////////////////////////////////////////////////////
    /// Copy constructor
    ///
    /// \param copy : instance to copy
    ///
    ////////////////////////////////////////////////////////////
    Image(const Image& copy);

    ////////////////////////////////////////////////////////////
    /// Destructor
    ///
    ////////////////////////////////////////////////////////////
    ~Image();

    ////////////////////////////////////////////////////////////
    /// Load the image from a file
    ///
    /// \param Filename : Path of the image file to load
    ///
    /// \return True if loading was successful
    ///
    ////////////////////////////////////////////////////////////
    bool LoadFromFile(const std::string& filename);

    ////////////////////////////////////////////////////////////
    /// Load the image from a file in memory
    ///
    /// \param data :        Pointer to the file data in memory
    /// \param sizeInBytes : Size of the data to load, in bytes
    ///
    /// \return True if loading was successful
    ///
    ////////////////////////////////////////////////////////////
    bool LoadFromMemory(const void* data, std::size_t sizeInBytes);

    ////////////////////////////////////////////////////////////
    /// Load the image directly from an array of pixels
    ///
    /// \param width :  Image width
    /// \param height : Image height
    /// \param pixels : Pointer to the pixels in memory (assumed format is RGBA)
    ///
    /// \return True if loading was successful
    ///
    ////////////////////////////////////////////////////////////
    bool LoadFromPixels(unsigned int width, unsigned int height, const Uint8* pixels);

    ////////////////////////////////////////////////////////////
    /// Save the content of the image to a file
    ///
    /// \param filename : Path of the file to save (overwritten if already exist)
    ///
    /// \return True if saving was successful
    ///
    ////////////////////////////////////////////////////////////
    bool SaveToFile(const std::string& filename) const;

    ////////////////////////////////////////////////////////////
    /// Create an empty image
    ///
    /// \param width :  Image width
    /// \param height : Image height
    /// \param color :  Image color
    ///
    /// \return True if creation was successful
    ///
    ////////////////////////////////////////////////////////////
    bool Create(unsigned int width, unsigned int height, const Color& color = Color(0, 0, 0));

    ////////////////////////////////////////////////////////////
    /// Create transparency mask from a specified colorkey
    ///
    /// \param transparentColor : Color to become transparent
    /// \param alpha :            Alpha value to assign to transparent pixels
    ///
    ////////////////////////////////////////////////////////////
    void CreateMaskFromColor(const Color& transparentColor, Uint8 alpha = 0);

    ////////////////////////////////////////////////////////////
    /// Copy pixels from another image onto this one.
    /// This function does a slow pixel copy and should only
    /// be used at initialization time
    ///
    /// \param source :     Source image to copy
    /// \param destX :      X coordinate of the destination position
    /// \param destY :      Y coordinate of the destination position
    /// \param sourceRect : Sub-rectangle of the source image to copy
    /// \param applyAlpha : Should the copy take in account the source transparency?
    ///
    ////////////////////////////////////////////////////////////
    void Copy(const Image& source, unsigned int destX, unsigned int destY, const IntRect& sourceRect = IntRect(0, 0, 0, 0), bool applyAlpha = false);

    ////////////////////////////////////////////////////////////
    /// Create the image from the current contents of the
    /// given window
    ///
    /// \param window :     Window to capture
    /// \param sourceRect : Sub-rectangle of the screen to copy (empty by default - entire image)
    ///
    /// \return True if copy was successful
    ///
    ////////////////////////////////////////////////////////////
    bool CopyScreen(RenderWindow& window, const IntRect& sourceRect = IntRect(0, 0, 0, 0));

    ////////////////////////////////////////////////////////////
    /// Change the color of a pixel
    ///
    /// \param x :     X coordinate of pixel in the image
    /// \param y :     Y coordinate of pixel in the image
    /// \param color : New color for pixel (x, y)
    ///
    ////////////////////////////////////////////////////////////
    void SetPixel(unsigned int x, unsigned int y, const Color& color);

    ////////////////////////////////////////////////////////////
    /// Get a pixel from the image
    ///
    /// \param x : X coordinate of pixel in the image
    /// \param y : Y coordinate of pixel in the image
    ///
    /// \return Color of pixel (x, y)
    ///
    ////////////////////////////////////////////////////////////
    const Color& GetPixel(unsigned int x, unsigned int y) const;

    ////////////////////////////////////////////////////////////
    /// Get a read-only pointer to the array of pixels (RGBA 8 bits integers components)
    /// Array size is GetWidth() x GetHeight() x 4
    /// This pointer becomes invalid if you reload or resize the image
    ///
    /// \return Const pointer to the array of pixels
    ///
    ////////////////////////////////////////////////////////////
    const Uint8* GetPixelsPtr() const;

    ////////////////////////////////////////////////////////////
    /// Update the whole image from an array of pixels
    ///
    /// \param pixels : Array of pixels to write to the image
    ///
    ////////////////////////////////////////////////////////////
    void UpdatePixels(const Uint8* pixels);

    ////////////////////////////////////////////////////////////
    /// Update a sub-rectangle of the image from an array of pixels
    ///
    /// Warning: for performances reasons, this function doesn't
    /// perform any check; thus you're responsible of ensuring that
    /// \a rectangle does not exceed the image size, and that
    /// \a pixels contains enough elements.
    ///
    /// \param rectangle : Sub-rectangle of the image to update
    /// \param pixels :    Array of pixels to write to the image
    ///
    ////////////////////////////////////////////////////////////
    void UpdatePixels(const Uint8* pixels, const IntRect& rectangle);

    ////////////////////////////////////////////////////////////
    /// Bind the image for rendering
    ///
    ////////////////////////////////////////////////////////////
    void Bind() const;

    ////////////////////////////////////////////////////////////
    /// Enable or disable image smooth filter.
    /// This parameter is enabled by default
    ///
    /// \param smooth : True to enable smoothing filter, false to disable it
    ///
    ////////////////////////////////////////////////////////////
    void SetSmooth(bool smooth);

    ////////////////////////////////////////////////////////////
    /// Return the width of the image
    ///
    /// \return Width in pixels
    ///
    ////////////////////////////////////////////////////////////
    unsigned int GetWidth() const;

    ////////////////////////////////////////////////////////////
    /// Return the height of the image
    ///
    /// \return Height in pixels
    ///
    ////////////////////////////////////////////////////////////
    unsigned int GetHeight() const;

    ////////////////////////////////////////////////////////////
    /// Tells whether the smooth filtering is enabled or not
    ///
    /// \return True if image smoothing is enabled
    ///
    ////////////////////////////////////////////////////////////
    bool IsSmooth() const;

    ////////////////////////////////////////////////////////////
    /// Convert a subrect expressed in pixels, into float
    /// texture coordinates
    ///
    /// \param rectangle : Sub-rectangle of image to convert
    ///
    /// \return Texture coordinates corresponding to the sub-rectangle
    ///
    ////////////////////////////////////////////////////////////
    FloatRect GetTexCoords(const IntRect& rectangle) const;

    ////////////////////////////////////////////////////////////
    /// Get the maximum image size according to hardware support
    ///
    /// \return Maximum size allowed for images, in pixels
    ///
    ////////////////////////////////////////////////////////////
    static unsigned int GetMaximumSize();

    ////////////////////////////////////////////////////////////
    /// Get a valid image size according to hardware support
    ///
    /// \param Size : size to convert
    ///
    /// \return Valid nearest size (greater than or equal to specified size)
    ///
    ////////////////////////////////////////////////////////////
    static unsigned int GetValidSize(unsigned int size);

    ////////////////////////////////////////////////////////////
    /// Assignment operator
    ///
    /// \param other : instance to assign
    ///
    /// \return Reference to the image
    ///
    ////////////////////////////////////////////////////////////
    Image& operator =(const Image& other);

private :

    friend class Renderer;
    friend class RenderImage;

    ////////////////////////////////////////////////////////////
    /// Create the OpenGL texture
    ///
    /// \return True if texture has been successfully created
    ///
    ////////////////////////////////////////////////////////////
    bool CreateTexture();

    ////////////////////////////////////////////////////////////
    /// Make sure the texture in video memory is updated with the
    /// array of pixels
    ///
    ////////////////////////////////////////////////////////////
    void EnsureTextureUpdate() const;

    ////////////////////////////////////////////////////////////
    /// Make sure the array of pixels is updated with the
    /// texture in video memory
    ///
    ////////////////////////////////////////////////////////////
    void EnsureArrayUpdate() const;

    ////////////////////////////////////////////////////////////
    /// Make sure that the image is ready to be used
    ///
    ////////////////////////////////////////////////////////////
    void Use() const;

    ////////////////////////////////////////////////////////////
    /// Reset the image attributes
    ///
    ////////////////////////////////////////////////////////////
    void Reset();

    ////////////////////////////////////////////////////////////
    /// Destroy the OpenGL texture
    ///
    ////////////////////////////////////////////////////////////
    void DestroyTexture();

    ////////////////////////////////////////////////////////////
    // Types
    ////////////////////////////////////////////////////////////
    typedef std::vector<Color> ColorArray; ///< Array of colors

    ////////////////////////////////////////////////////////////
    // Member data
    ////////////////////////////////////////////////////////////
    unsigned int       myWidth;          ///< Image width
    unsigned int       myHeight;         ///< Image Height
    unsigned int       myTextureWidth;   ///< Actual texture width (can be greater than image width because of padding)
    unsigned int       myTextureHeight;  ///< Actual texture height (can be greater than image height because of padding)
    unsigned int       myTexture;        ///< Internal texture identifier
    bool               myIsSmooth;       ///< Status of the smooth filter
    mutable ColorArray myPixels;         ///< Pixels of the image
    mutable bool       myTextureUpdated; ///< Status of synchronization between pixels in central memory and the internal texture un video memory
    mutable bool       myArrayUpdated;   ///< Status of synchronization between pixels in central memory and the internal texture un video memory
    mutable bool       myPixelsFlipped;  ///< To work around the inconsistency in Y orientation
};

} // namespace sf


#endif // SFML_IMAGE_HPP
