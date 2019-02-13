﻿using System;
using Microsoft.ML.Data;
using Microsoft.ML.LightGBM;
using static Microsoft.ML.LightGBM.Options;

namespace Microsoft.ML.Samples.Dynamic
{
    class LightGbmRegressionWithOptions
    {
        /// <summary>
        /// This example require installation of addition nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.LightGBM/">Microsoft.ML.LightGBM</a>
        /// </summary>
        public static void Example()
        {
            // Downloading a regression dataset from github.com/dotnet/machinelearning
            // this will create a housing.txt file in the filsystem this code will run
            // you can open the file to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var mlContext = new MLContext();

            // Creating a data reader, based on the format of the data
            // The data is tab separated with all numeric columns.
            // The first column being the label and rest are numeric features
            // Here only seven numeric columns are used as features
            var dataView = mlContext.Data.ReadFromTextFile(dataFile, new TextLoader.Arguments
            {
                Separators = new[] { '\t' },
                HasHeader = true,
                Columns = new[]
               {
                    new TextLoader.Column("LabelColumn", DataKind.R4, 0),
                    new TextLoader.Column("FeaturesColumn", DataKind.R4, 1, 6)
                }
            });

            //////////////////// Data Preview ////////////////////
            // MedianHomeValue    CrimesPerCapita    PercentResidental    PercentNonRetail    CharlesRiver    NitricOxides    RoomsPerDwelling    PercentPre40s
            // 24.00              0.00632            18.00                2.310               0               0.5380          6.5750              65.20
            // 21.60              0.02731            00.00                7.070               0               0.4690          6.4210              78.90
            // 34.70              0.02729            00.00                7.070               0               0.4690          7.1850              61.10

            var (trainData, testData) = mlContext.Regression.TrainTestSplit(dataView, testFraction: 0.1);

            // Create a pipeline with LightGbm estimator with advanced options,
            // here we only need LightGbm trainer as data is already processed
            // in a form consumable by the trainer
            var options = new Options
            {
                LabelColumn = "LabelColumn",
                FeatureColumn = "FeaturesColumn",
                NumLeaves = 4,
                MinDataPerLeaf = 6,
                LearningRate = 0.001,
                Booster = new GossBooster.Arguments
                {
                    TopRate = 0.3,
                    OtherRate = 0.2
                }
            };
            var pipeline = mlContext.Regression.Trainers.LightGbm(options);

            // Fit this pipeline to the training data
            var model = pipeline.Fit(trainData);

            // Get the feature importance based on the information gain used during training.
            VBuffer<float> weights = default;
            model.Model.GetFeatureWeights(ref weights);
            var weightsValues = weights.GetValues();
            Console.WriteLine($"weight 0 - {weightsValues[0]}"); // CrimesPerCapita  (weight 0) = 0.1898361
            Console.WriteLine($"weight 5 - {weightsValues[5]}"); // RoomsPerDwelling (weight 5) = 1

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(dataWithPredictions, "LabelColumn");
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Output
            // L1: 4.97
            // L2: 51.37
            // LossFunction: 51.37
            // RMS: 7.17
            // RSquared: 0.08
        }
    }
}
