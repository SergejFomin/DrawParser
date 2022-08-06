using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DrawParser.View
{
    internal static class BitmapHelper
    {
        internal static Bitmap CropBitmap(Bitmap bitmap, int width, int height)
        {
            int? lowestx = null;
            int? lowesty = null;
            int? highestx = null;
            int? highesty = null;

            for (int x = 0; x < bitmap.Width; x++)
            {
                for (int y = 0; y < bitmap.Height; y++)
                {
                    var color = bitmap.GetPixel(x, y);
                    if (color.ToArgb() != System.Drawing.Color.White.ToArgb())
                    {
                        if (!lowestx.HasValue || x < lowestx)
                        {
                            lowestx = x;
                        }
                        if (!lowesty.HasValue || y < lowesty)
                        {
                            lowesty = y;
                        }
                        if (!highestx.HasValue || x > highestx)
                        {
                            highestx = x;
                        }
                        if (!highesty.HasValue || y > highesty)
                        {
                            highesty = y;
                        }
                    }
                }
            }

            if (lowestx.HasValue && lowesty.HasValue && highestx.HasValue && highesty.HasValue)
            {
                int margin = 5;

                var size = new System.Drawing.Size(highestx.Value - lowestx.Value + 1, highesty.Value - lowesty.Value + 1);
                if (size.Width < size.Height)
                {
                    // diff in side length
                    var diff = size.Height - size.Width;

                    // adding/subtracting half the diff to the 2 bound
                    lowestx -= diff / 2;
                    highestx += diff / 2;

                    if (lowestx < 0)
                    {
                        highestx += lowestx * (-1);
                        lowestx = 0;
                    }

                    if (highestx > width)
                    {
                        highestx = width;
                    }
                }
                else
                {
                    // diff in side length
                    var diff = size.Width - size.Height;

                    // adding/subtracting half the diff to the 2 bound
                    lowesty -= diff / 2;
                    highesty += diff / 2;

                    if (lowesty < 0)
                    {
                        highesty += lowesty * (-1);
                        lowesty = 0;
                    }

                    if (highesty > height)
                    {
                        highesty = height;
                    }
                }

                lowestx = lowestx - margin >= 0 ? lowestx - margin : lowestx;
                lowesty = lowesty - margin >= 0 ? lowesty - margin : lowesty;
                highestx = highestx + margin <= width ? highestx + margin : highestx;
                highesty = highesty + margin <= height ? highesty + margin : highesty;
                var normedSize = new System.Drawing.Size(highestx.Value - lowestx.Value, highesty.Value - lowesty.Value);

                var rectangle = new System.Drawing.Rectangle(lowestx.Value, lowesty.Value, normedSize.Width, normedSize.Height);
                Bitmap croppedBitmap = bitmap.Clone(rectangle, bitmap.PixelFormat);
                return croppedBitmap;
            }

            return bitmap;
        }

        internal static Bitmap MakeGrayscale(Bitmap original)
        {
            //create a blank bitmap the same size as original
            Bitmap newBitmap = new Bitmap(original.Width, original.Height);

            //get a graphics object from the new image
            using (Graphics g = Graphics.FromImage(newBitmap))
            {

                //create the grayscale ColorMatrix
                ColorMatrix colorMatrix = new ColorMatrix(
                   new float[][]
                   {
                        new float[] {.3f, .3f, .3f, 0, 0},
                        new float[] {.59f, .59f, .59f, 0, 0},
                        new float[] {.11f, .11f, .11f, 0, 0},
                        new float[] {0, 0, 0, 1, 0},
                        new float[] {0, 0, 0, 0, 1}
                   });

                //create some image attributes
                using (ImageAttributes attributes = new ImageAttributes())
                {

                    //set the color matrix attribute
                    attributes.SetColorMatrix(colorMatrix);

                    //draw the original image on the new image
                    //using the grayscale color matrix
                    g.DrawImage(original, new System.Drawing.Rectangle(0, 0, original.Width, original.Height),
                                0, 0, original.Width, original.Height, GraphicsUnit.Pixel, attributes);
                }
            }
            return newBitmap;
        }

        internal static Bitmap ParseFromDarkness(float[] darknessValues)
        {
            throw new NotImplementedException();
        }
    }
}
