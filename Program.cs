using System;
using System.IO;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using YOLOv4MLNet.DataStructures;
using ImageClassification;
using ImageDatabase;
using System.Drawing;

class Program
{
    static async Task Main(string[] args)
    {
        if (args.Length != 1)
        {
            Console.Error.WriteLine("Usage: %program_name% imageDirectory");
            Environment.Exit(1);
        }
        string dir = args[0];
        ImageClassifier imageClassifierModel = new ImageClassifier(dir);

        using (var db = new DatabaseStoreContext())
        {
            await foreach (Tuple<string, List<YoloV4Result>> imgRes in
                           imageClassifierModel.ProcessDirectoryContentsAsync())
            {
                ProcessedImage processedImage = new ProcessedImage(imgRes.Item1);

                foreach (YoloV4Result res in imgRes.Item2)
                {
                    // x1, y1, x2, y2 in page coordinates.
                    // left, top, right, bottom.
                    float x1 = res.BBox[0];
                    float y1 = res.BBox[1];
                    float x2 = res.BBox[2];
                    float y2 = res.BBox[3];
                    string label = res.Label;
                    RecognizedObject recObj = new RecognizedObject(x1, y1, x2, y2, label);
                    processedImage.RecognizedObjects.Add(recObj);
                }
                db.AddProcessedImage(processedImage);
                db.SaveChanges();
            }
        }
    }
}
