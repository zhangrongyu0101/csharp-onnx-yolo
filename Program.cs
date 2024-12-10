using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace OnnxYoloV5
{
    class Program
        {

        const string modelPath = @"Assets\Models\yolov5s_tire.onnx";

        const string imageFolder = @"Assets\Images";

        const string imageOutputFolder = @"Assets\Output";
        static readonly string[] classesNames = new string[] 
        { 
            "0", "1", "2", "3", "4", "6", "7", "8", "A", "B", "E", "G", "H", "Y" 
        };

        public static Bitmap ConvertBGRToBitmap(byte[] bgrPixels, int width, int height)
        {

            if (bgrPixels.Length != width * height * 3)
                throw new ArgumentException("Byte array size does not match the image dimensions.");

            Bitmap bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);

            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bitmap.PixelFormat);

            IntPtr ptr = bitmapData.Scan0;

            // Copy the BGR byte array to the bitmap's pixel data
            Marshal.Copy(bgrPixels, 0, ptr, bgrPixels.Length);

            // Unlock the bits after copying the data
            bitmap.UnlockBits(bitmapData);

            return bitmap;
        }

        static void Main(string[] args)
        {
            
            Directory.CreateDirectory(imageOutputFolder);
            MLContext mlContext = new MLContext();
            var pipeline = mlContext.Transforms.ResizeImages(
                                                                inputColumnName: "images", 
                                                                outputColumnName: "images", 
                                                                imageWidth: 640, 
                                                                imageHeight: 640, 
                                                                resizing: ResizingKind.Fill)
                .Append(mlContext.Transforms.ExtractPixels(
                                                            outputColumnName: "images", 
                                                            scaleImage: 1f / 255f, 
                                                            interleavePixelColors: false))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                            outputColumnNames: new[] { "output0" },
                            inputColumnNames: new[] { "images" },
                            modelFile: modelPath,
                            
                            shapeDictionary: new Dictionary<string, int[]>
                            {
                                { "images", new[] { 1, 3, 640, 640 } },
                                { "output0", new[] { 1, 25200, 19 } }
                            },
                            gpuDeviceId: null, 
                            fallbackToCpu: true
                        ));


            List<YoloV5BitmapData> emptyData = new List<YoloV5BitmapData>();
            IDataView emptyDataView = mlContext.Data.LoadFromEnumerable(emptyData);
            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(emptyDataView);

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV5BitmapData, YoloV5Prediction>(model);

            // save model
            //mlContext.Model.Save(model, predictionEngine.OutputSchema, Path.ChangeExtension(modelPath, "zip"));

            foreach (string imageName in Directory.GetFiles(imageFolder))
            {

                using (var img = MLImage.CreateFromFile(imageName))
                {
                    // predict
                    var predict = predictionEngine.Predict(new YoloV5BitmapData() { Image = img });
                    var results = predict.GetResults(classesNames, 0.3f, 0.7f);

                    byte[] getBGRPixels = img.GetBGRPixels;

                    var bitmap = ConvertBGRToBitmap(img.GetBGRPixels, img.Width, img.Height);

                    using (var g = Graphics.FromImage(bitmap))
                    {
                        foreach (var res in results)
                        {
                            // draw predictions
                            var x1 = res.BBox[0];
                            var y1 = res.BBox[1];
                            var x2 = res.BBox[2];
                            var y2 = res.BBox[3];
                            g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                            using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                            {
                                g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                            }

                            g.DrawString(res.Label + " " + res.Confidence.ToString("0.00"),
                                         new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                        }
                        var ss = Path.Combine(imageOutputFolder, Path.GetFileNameWithoutExtension(imageName) + "_Processed" + Path.GetExtension(imageName));
                        bitmap.Save(ss);
                        Console.WriteLine(Path.GetFileNameWithoutExtension(imageName) + " Processed ");
                    }
                }
            }
        }
    }
}
