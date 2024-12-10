using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace OnnxYoloV5
{
    class YoloV5BitmapData
    {
        [ColumnName("images")]
        [ImageType(640, 640)]
        public MLImage Image { get; set; }

        [ColumnName("width")]
        public float ImageWidth => Image.Width;

        [ColumnName("height")]
        public float ImageHeight => Image.Height;
    }
}
