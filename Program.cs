using System;
using System.Drawing;
using Microsoft.ML;
using Microsoft.ML.Data;

// Define the input and output data classes
public class FaceMaskInput
{
    [ColumnName("bitmap"), ImageType(224, 224)]
    public Bitmap Image { get; set; }
}

public class FaceMaskOutput
{
    [ColumnName("softmax2"), VectorType(2)]
    public float[] Prediction { get; set; }
}

class Program2
{
    static void Main(string[] args)
    {
        // Create a new MLContext
        var mlContext = new MLContext();

        // Load the pre-trained model from a file
        var model = mlContext.Model.Load("model.zip", out var modelSchema);

        // Create a prediction engine
        var engine = mlContext.Model.CreatePredictionEngine<FaceMaskInput, FaceMaskOutput>(model);

        // Load an image
        var image = new Bitmap("test.jpg");

        // Make a prediction
        var input = new FaceMaskInput { Image = image };
        var output = engine.Predict(input);

        // Display the prediction
        var prediction = output.Prediction;
        var isWearingMask = prediction[1] > prediction[0]; // Check if the "mask" class has a higher probability
        Console.WriteLine($"The person {(isWearingMask ? "is wearing" : "is not wearing")} a mask.");
    }
}
