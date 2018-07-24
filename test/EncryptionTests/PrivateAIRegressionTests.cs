// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using System;
using System.IO;
using Xunit;
using Microsoft.Research.SEAL;
using Microsoft.ML.Runtime.Training;

namespace EncryptionTests
{
    public partial class PrivateAITests
    {
        [Fact]
        public void EncryptedScorerRegressionTest()
        {
            string dataPath = @"E:\Experiments\UCIHousing\housing.txt";  // Data is not in our repo currently
            var modelFile = @"\\ct01\users\zeahmed\ML.NET\build\Housing.zip";

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Train the model and save it in modelFile
                TrainRegressionModel(env, dataPath, modelFile);
                RunTest(env, dataPath, modelFile);
            }
        }

        private void TrainRegressionModel(TlcEnvironment env,string dataPath, string modelFile)
        {
            // Pipeline
            var loader = new TextLoader(env,
                new TextLoader.Arguments()
                {
                    HasHeader = false,
                    Column = new[] {
                            new TextLoader.Column()
                            {
                                Name = "Label",
                                Source = new [] { new TextLoader.Range() { Min = 0, Max = 0} },
                                Type = DataKind.R4
                            },
                            new TextLoader.Column()
                            {
                                Name = "Features",
                                Source = new [] { new TextLoader.Range() { Min = 1, Max = 13} },
                                Type = DataKind.R4
                            }
                    }
                }, new MultiFileSource(dataPath));

            // Normalizer is not automatically added though the trainer has 'NormalizeFeatures' On/Auto
            var trans = NormalizeTransform.CreateMinMaxNormalizer(env, loader, "Features");

            // Train
            var trainer = new SdcaRegressionTrainer(env, new SdcaRegressionTrainer.Arguments() { NumThreads = 1 });
            // Explicity adding CacheDataView since caching is not working though trainer has 'Caching' On/Auto
            var cached = new CacheDataView(env, trans, prefetch: null);
            var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
            var pred = (LinearRegressionPredictor)trainer.Train(trainRoles);

            // Set the Evaluator that will be used for computing
            pred.Evaluator = EncryContext.Evaluator;

            // Now encrypt the model
            pred.EncryptModel(EncryContext.Encryptor, EncryContext.Encoder);

            // Save model
            trainRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
            SaveModel(env, pred, trainRoles, modelFile);
        }
    }
}
