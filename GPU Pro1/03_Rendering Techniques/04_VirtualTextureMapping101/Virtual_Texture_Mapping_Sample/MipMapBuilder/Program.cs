/*
	Virtual texture mapping demo app
    Copyright (C) 2008, 2009 Matthäus G. Chajdas
    Contact: shaderx8@anteru.net

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

using System;
using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommandLine;
using CommandLine.Text;

namespace MipMapBuilder
{
    class Program
    {
        internal sealed class Options
        {
            [Option("i", "input",
                Required = true)]
            public string input = "tile-";

            [HelpOption(HelpText = "Display this help screen")]
            public string GetUsage()
            {
                HelpText ht = new HelpText("Mipmap Generator");
                ht.Copyright = new CopyrightInfo("Matthaeus G. Chajdas", 2008);
                ht.AddOptions(this);

                return ht;
            }
        }

        static void TileImage(BitmapSource image, int level)
        {
            // Chop the image into 128² large tiles, then add borders of 4 pixels

            // Compute the number of tiles
            int tileCount = image.PixelWidth / 128;
            PixelFormat format = image.Format;
            int bytesPerPixel = format.BitsPerPixel / 8;

            for (int y = 0; y < tileCount; ++y)
            {
                for (int x = 0; x < tileCount; ++x)
                {
                    System.Console.WriteLine("Processing: Level {0}, Tile {1}/{2}", level, y * tileCount + x, tileCount * tileCount);
                    WriteableBitmap wb = new WriteableBitmap(136, 136, 96, 96, format, null);

                    byte[] array = new byte[128 * 128 * bytesPerPixel];
                    image.CopyPixels(new System.Windows.Int32Rect(x * 128, y * 128, 128, 128), array, 128 * bytesPerPixel, 0);

                    wb.WritePixels(new System.Windows.Int32Rect(4, 4, 128, 128), array, 128 * bytesPerPixel, 0);

                    // Now, fill the rest
                    // Left side
                    if (x == 0)
                    {
                        // Duplicate the leftmost pixels 4 times
                        byte[] tmp = new byte[128 * bytesPerPixel];
                        wb.CopyPixels(new System.Windows.Int32Rect(4, 4, 1, 128), tmp, bytesPerPixel, 0);
                        for (int i = 0; i < 4; ++i)
                        {
                            wb.WritePixels(new System.Windows.Int32Rect(i, 4, 1, 128), tmp, bytesPerPixel, 0);
                        }
                    }
                    else
                    {
                        // Copy from the image  to the left
                        byte[] tmp = new byte[128 * 4 * bytesPerPixel];
                        image.CopyPixels(new System.Windows.Int32Rect(x * 128 - 4, y * 128, 4, 128), array, 4 * bytesPerPixel, 0);
                        wb.WritePixels(new System.Windows.Int32Rect(0, 4, 4, 128), array, 4 * bytesPerPixel, 0);
                    }
                    
                    // Right side
                    if (x == (tileCount - 1))
                    {
                        // Duplicate the rightmost pixels 4 times
                        byte[] tmp = new byte[128 * bytesPerPixel];
                        wb.CopyPixels(new System.Windows.Int32Rect(4 + 127, 4, 1, 128), tmp, bytesPerPixel, 0);
                        for (int i = 0; i < 4; ++i)
                        {
                            wb.WritePixels(new System.Windows.Int32Rect(4 + 128 + i, 4, 1, 128), tmp, bytesPerPixel, 0);
                        }
                    }
                    else
                    {
                        // Copy from the image  to the right
                        byte[] tmp = new byte[128 * 4 * bytesPerPixel];
                        image.CopyPixels(new System.Windows.Int32Rect(x * 128 + 128, y * 128, 4, 128), array, 4 * bytesPerPixel, 0);
                        wb.WritePixels(new System.Windows.Int32Rect(128 + 4, 4, 4, 128), array, 4 * bytesPerPixel, 0);
                    }

                    // Top side
                    if (y == 0)
                    {
                        // Duplicate topmost pixels
                        byte[] tmp = new byte[128 * bytesPerPixel];
                        wb.CopyPixels(new System.Windows.Int32Rect(4, 4, 128, 1), tmp, 128 * bytesPerPixel, 0);
                        for (int i = 0; i < 4; ++i)
                        {
                            wb.WritePixels(new System.Windows.Int32Rect(4, i, 128, 1), tmp, 128 * bytesPerPixel, 0);
                        }
                    }
                    else
                    {
                        // Copy pixels from above
                        byte[] tmp = new byte[128 * 4 * bytesPerPixel];
                        image.CopyPixels(new System.Windows.Int32Rect(x * 128, y * 128 - 4, 128, 4), tmp, 128 * bytesPerPixel, 0);
                        wb.WritePixels(new System.Windows.Int32Rect(4, 0, 128, 4), tmp, 128 * bytesPerPixel, 0);
                    }                    // Top side
                    
                    if (y == (tileCount - 1))
                    {
                        // Duplicate bottom pixels
                        byte[] tmp = new byte[128 * bytesPerPixel];
                        wb.CopyPixels(new System.Windows.Int32Rect(4, 4 + 127, 128, 1), tmp, 128 * bytesPerPixel, 0);
                        for (int i = 0; i < 4; ++i)
                        {
                            wb.WritePixels(new System.Windows.Int32Rect(4, 4 + 128 + i, 128, 1), tmp, 128 * bytesPerPixel, 0);
                        }
                    }
                    else
                    {
                        // Copy pixels from below
                        byte[] tmp = new byte[128 * 4 * bytesPerPixel];
                        image.CopyPixels(new System.Windows.Int32Rect(x * 128, y * 128 + 128, 128, 4), tmp, 128 * bytesPerPixel, 0);
                        wb.WritePixels(new System.Windows.Int32Rect(4, 4 + 128, 128, 4), tmp, 128 * bytesPerPixel, 0);
                    }

                    if (x == 0 || y == 0)
                    {
                        // Top-Left Corner
                        byte[] tmp = new byte[bytesPerPixel];
                        wb.CopyPixels(new System.Windows.Int32Rect(4, 4, 1, 1), tmp, bytesPerPixel, 0);

                        for (int i = 0; i < 4; ++i)
                        {
                            for (int j = 0; j < 4; ++j)
                            {
                                wb.WritePixels(new System.Windows.Int32Rect(i, j, 1, 1), tmp, bytesPerPixel, 0);
                            }
                        }
                    }
                    else
                    {
                        byte[] tmp = new byte[4 * 4 * bytesPerPixel];
                        image.CopyPixels(new System.Windows.Int32Rect(x * 128 - 4, y * 128 - 4, 4, 4), tmp, 4 * bytesPerPixel, 0);
                        wb.WritePixels(new System.Windows.Int32Rect(0, 0, 4, 4), tmp, 4 * bytesPerPixel, 0);
                    }
                    
                    if (x == 0 || y == (tileCount - 1))
                    {
                        // Bottom left
                        byte[] tmp = new byte[bytesPerPixel];
                        wb.CopyPixels(new System.Windows.Int32Rect(4, 4 + 127, 1, 1), tmp, bytesPerPixel, 0);

                        for (int i = 0; i < 4; ++i)
                        {
                            for (int j = 0; j < 4; ++j)
                            {
                                wb.WritePixels(new System.Windows.Int32Rect(i, 128 + 4 + j, 1, 1), tmp, bytesPerPixel, 0);
                            }
                        }
                    }
                    else
                    {
                        byte[] tmp = new byte[4 * 4 * bytesPerPixel];
                        image.CopyPixels(new System.Windows.Int32Rect(x * 128 - 4, y * 128 + 128, 4, 4), tmp, 4 * bytesPerPixel, 0);
                        wb.WritePixels(new System.Windows.Int32Rect(0, 128 + 4, 4, 4), tmp, 4 * bytesPerPixel, 0);
                    }

                    if (x == (tileCount-1) || y == 0)
                    {
                        // Top-Right Corner
                        byte[] tmp = new byte[bytesPerPixel];
                        wb.CopyPixels(new System.Windows.Int32Rect(4 + 127, 4, 1, 1), tmp, bytesPerPixel, 0);

                        for (int i = 0; i < 4; ++i)
                        {
                            for (int j = 0; j < 4; ++j)
                            {
                                wb.WritePixels(new System.Windows.Int32Rect(128 + 4 + i, j, 1, 1), tmp, bytesPerPixel, 0);
                            }
                        }
                    }
                    else
                    {
                        byte[] tmp = new byte[4 * 4 * bytesPerPixel];
                        image.CopyPixels(new System.Windows.Int32Rect(x * 128 + 128, y * 128, 4, 4), tmp, 4 * bytesPerPixel, 0);
                        wb.WritePixels(new System.Windows.Int32Rect(128 + 4, 0, 4, 4), tmp, 4 * bytesPerPixel, 0);
                    }

                    if (x == (tileCount-1) || y == (tileCount - 1))
                    {
                        // Bottom right
                        byte[] tmp = new byte[bytesPerPixel];
                        wb.CopyPixels(new System.Windows.Int32Rect(4 + 127, 4 + 127, 1, 1), tmp, bytesPerPixel, 0);

                        for (int i = 0; i < 4; ++i)
                        {
                            for (int j = 0; j < 4; ++j)
                            {
                                wb.WritePixels(new System.Windows.Int32Rect(128 + 4 + i, 128 + 4 + j, 1, 1), tmp, bytesPerPixel, 0);
                            }
                        }
                    }
                    else
                    {
                        byte[] tmp = new byte[4 * 4 * bytesPerPixel];
                        image.CopyPixels(new System.Windows.Int32Rect(x * 128 + 128, y * 128 + 128, 4, 4), tmp, 4 * bytesPerPixel, 0);
                        wb.WritePixels(new System.Windows.Int32Rect(128 + 4, 128 + 4, 4, 4), tmp, 4 * bytesPerPixel, 0);
                    }

                    JpegBitmapEncoder encoder = new JpegBitmapEncoder();
                    encoder.QualityLevel = 95;
                    BitmapFrame bf = BitmapFrame.Create(wb);

                    encoder.Frames.Add(bf);
                    encoder.Save(File.Create(String.Format("tile-{0}-{1}.jpg", level, y * tileCount + x)));
                }
            }
        }

        static void Main(string[] args)
        {
            Options o = new Options();

            bool r = CommandLine.Parser.ParseArguments(args, o, System.Console.Out);

            if (!r)
            {
                return;
            }

            
            for (int i = 0; i <= 6; ++i)
            {
                Stream source = File.Open(o.input, FileMode.Open, FileAccess.Read);
                BitmapImage img = new BitmapImage();
                img.BeginInit();
                img.DecodePixelWidth = 8192 / (1 << i);
                img.DecodePixelHeight = 8192 / (1 << i);
                img.StreamSource = source;
                img.EndInit();

                TileImage(img, i);
                source.Close();
            }

        }
    }
}
