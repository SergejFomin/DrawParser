using DrawParser.View;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WpfAppdrawtest
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private MainWindowViewModel viewModel;

        public MainWindow()
        {
            InitializeComponent();
            this.canvas.Background = new SolidColorBrush(System.Windows.Media.Colors.White);
            this.viewModel = new MainWindowViewModel();
            this.viewModel.ClearCanvasAction = this.canvas.Strokes.Clear;
            this.DataContext = this.viewModel;

            this.canvas.Strokes.StrokesChanged += this.CanvasStrokesChanged;
        }

        private void CanvasStrokesChanged(object sender, System.Windows.Ink.StrokeCollectionChangedEventArgs e)
        {
            this.UpdateImageAndData();
        }

        private void ClearCanvas_Click(object sender, RoutedEventArgs e)
        {
            this.canvas.Strokes.Clear();
        }

        private void UpdateImageAndData()
        {
            Bitmap resultBitmap;
            RenderTargetBitmap renderTargetBitmap = new RenderTargetBitmap((int)this.canvas.ActualWidth, (int)this.canvas.ActualHeight, 96d, 96d, PixelFormats.Default);
            renderTargetBitmap.Render(this.canvas);

            //save the ink to a memory stream
            BmpBitmapEncoder encoder = new BmpBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(renderTargetBitmap));

            using (MemoryStream ms = new MemoryStream())
            {
                encoder.Save(ms);

                //get the bitmap bytes from the memory stream
                ms.Position = 0;
                resultBitmap = BitmapHelper.CropBitmap(new Bitmap(ms), (int)this.canvas.ActualWidth, (int)this.canvas.ActualHeight);
            }

            resultBitmap = new Bitmap(resultBitmap, 28, 28);
            resultBitmap = BitmapHelper.MakeGrayscale(resultBitmap);

            this.SetBitmapInImage(resultBitmap);
            this.SetBitmapInDataSource(resultBitmap);
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            base.OnClosing(e);
            this.viewModel.OnClosing();
        }

        private void SetBitmapInImage(Bitmap bitmap)
        {
            BitmapImage bitmapImage = new BitmapImage();
            bitmapImage.BeginInit();

            var ms = new MemoryStream();
            bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
            ms.Seek(0, SeekOrigin.Begin);
            bitmapImage.StreamSource = ms;

            bitmapImage.EndInit();
            this.previewImage.Source = bitmapImage;
        }

        private void SetBitmapInDataSource(Bitmap bitmap)
        {
            var result = new double[bitmap.Width * bitmap.Height];
            for (int y = 0; y < bitmap.Width; y++)
            {
                for (int x = 0; x < bitmap.Height; x++)
                {
                    int index = y * bitmap.Width + x;
                    result[index] = (double)1 - bitmap.GetPixel(x, y).GetBrightness();
                }
            }

            this.viewModel.ImageBrightnessData = result;
        }
    }
}
