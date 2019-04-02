﻿using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ProduceWordBags
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "This is an example to compute bag-of-word features." },
                new TextData(){ Text = "ML.NET's ProduceWordBags API produces bag-of-word features from input text." },
                new TextData(){ Text = "It does so by first tokenizing text/string into words/tokens then " },
                new TextData(){ Text = "computing Ngram and their neumeric values." },
                new TextData(){ Text = "Each position in the output vector corresponds to a particular Ngram." },
                new TextData(){ Text = "The value at each position corresponds to," },
                new TextData(){ Text = "the number of times Ngram occured in the data (Tf), or" },
                new TextData(){ Text = "the inverse of the number of documents contain the Ngram (Idf), or." },
                new TextData(){ Text = "or compute both and multipy together (Tf-Idf)." },
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for converting text into numeric bag-of-word features.
            // The following call to 'ProduceWordBags' implicitly tokenizes the text/string into words/tokens.
            // Please note that the length of the output feature vector depends on the Ngram settings.
            var textPipeline = mlContext.Transforms.Text.ProduceWordBags("BagOfWordFeatures", "Text",
                ngramLength: 3, useAllLengths: false, weighting: NgramExtractingEstimator.WeightingCriteria.Tf);

            // Fit to data.
            var textTransformer = textPipeline.Fit(dataview);
            var transformedDataView = textTransformer.Transform(dataview);

            // Create the prediction engine to get the bag-of-word features extracted from the text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer);

            // Convert the text into numeric features.
            var prediction = predictionEngine.Predict(samples[0]);

            // Print the length of the feature vector.
            Console.WriteLine($"Number of Features: {prediction.BagOfWordFeatures.Length}");

            // Preview of the produced Ngrams.
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            transformedDataView.Schema["BagOfWordFeatures"].GetSlotNames(ref slotNames);
            var BagOfWordFeaturesColumn = transformedDataView.GetColumn<VBuffer<float>>(transformedDataView.Schema["BagOfWordFeatures"]);
            var slots = slotNames.GetValues();
            Console.Write("Ngrams: ");
            foreach (var featureRow in BagOfWordFeaturesColumn)
            {
                foreach (var item in featureRow.Items())
                    Console.Write($"{slots[item.Key]}  ");
                Console.WriteLine();
            }

            // Print the first 10 feature values.
            Console.Write("Features: ");
            for (int i = 0; i < 10; i++)
                Console.Write($"{prediction.BagOfWordFeatures[i]:F4}  ");

            //  Expected output:
            //   Number of Features: 62
            //   Ngrams:   This|is|an  is|an|example  an|example|to  example|to|compute  to|compute|bag-of-word  compute|bag-of-word|features.  ML.NET's|ProduceWordBags|API  ProduceWordBags|API|produces  API|produces|bag-of-word  produces|bag-of-word|features  ...
            //   Features:    1.0000       1.0000        1.0000           1.0000                1.0000                      1.0000                           0.0000                         0.0000                  0.0000                       0.0000              ...
        }

        public class TextData
        {
            public string Text { get; set; }
        }

        public class TransformedTextData : TextData
        {
            public float[] BagOfWordFeatures { get; set; }
        }
    }
}
