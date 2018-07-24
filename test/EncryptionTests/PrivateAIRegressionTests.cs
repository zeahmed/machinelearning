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
            

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Train the model, get the predictor and optionally get the test IDataView
                // to test our model on.
                IDataView testData = null;
                LinearRegressionPredictor pred = TrainRegressionModel(env, ref testData);

                // Get the valuemapper methods. Both for normal and encrypted case.
                // We will use these mappers to score the feature vector before and after encryption.
                // Since non of ML.Net transforms are encryption aware, feature vector is featurized here.
                // Featurized vector is then ecrypted and passed on to model for scoring.
                var valueMapperEncrypted = pred.GetEncryptedMapper<VBuffer<Ciphertext>, Ciphertext>();
                var valueMapper = pred.GetMapper<VBuffer<Single>, Single>();


                // Prepare for iteration over the data pipeline.
                var cursorFactory = new FloatLabelCursor.Factory(new RoleMappedData(testData, DefaultColumnNames.Label, DefaultColumnNames.Features)
                                                        , CursOpt.Label | CursOpt.Features);
                using (var cursor = cursorFactory.Create())
                {
                    double executionTime = 0;
                    double encryptedExecutionTime = 0;
                    int sampleCount = 0;
                    // Iterate over the data and match encrypted and non-encrypted score.
                    while (cursor.MoveNext())
                    {
                        sampleCount++;
                        // Predict on Encrypted Data
                        var vBufferencryptedFeatures = EncryptData(ref cursor.Features);
                        Ciphertext encryptedResult = new Ciphertext();
                        var watch = System.Diagnostics.Stopwatch.StartNew();
                        valueMapperEncrypted(ref vBufferencryptedFeatures, ref encryptedResult);

                        // Decode the encrypted prediction obtrained from the model.
                        var plainResult = new Plaintext();
                        EncryContext.Decryptor.Decrypt(encryptedResult, plainResult);
                        var predictionEncrypted = (float)EncryContext.Encoder.Decode(plainResult);
                        encryptedExecutionTime += watch.ElapsedTicks / 10000.0;
                        watch.Stop();


                        // Predict on non-ecrypted data.
                        float prediction = 0;
                        watch = System.Diagnostics.Stopwatch.StartNew();
                        valueMapper(ref cursor.Features, ref prediction);
                        executionTime += watch.ElapsedTicks / 10000.0;
                        watch.Stop();

                        // Compare the results to some tolerance.
                        Assert.True(Math.Abs(predictionEncrypted - prediction) <= (1e-05 + 1e-08 * Math.Abs(prediction)));
                    }

                    Output.WriteLine("Avg. Prediction Time : {0}ms", executionTime / sampleCount);
                    Output.WriteLine("Avg. Prediction Time (Encrypted) : {0}ms", encryptedExecutionTime / sampleCount);
                }
            }
        }

        private LinearRegressionPredictor TrainRegressionModel(TlcEnvironment env, ref IDataView testData)
        {
            string dataPath = @"E:\Experiments\UCIHousing\housing.txt";  // Data is not in our repo currently
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

            var modelFile = @"\\ct01\users\zeahmed\ML.NET\build\Housing.zip";
            // Save model
            SaveModel(env, pred, trainRoles, modelFile);
            pred = (LinearRegressionPredictor)LoadModel(env, modelFile);

            // Set the Evaluator after the model is loaded that will be used for computing
            pred.Evaluator = EncryContext.Evaluator;

            // Get test data. We are testing on the same file used for training.
            testData = GetTestPipelineRegression(env, trans, pred);
            return pred;
        }

        private IDataView GetTestPipelineRegression(IHostEnvironment env, IDataView transforms, IPredictor pred)
        {
            string testDataPath = @"E:\Experiments\UCIHousing\housing.txt";
            using (var ch = env.Start("Saving model"))
            using (var memoryStream = new MemoryStream())
            {
                var trainRoles = new RoleMappedData(transforms, label: "Label", feature: "Features");

                // Model cannot be saved with CacheDataView
                TrainUtils.SaveModel(env, ch, memoryStream, pred, trainRoles);
                memoryStream.Position = 0;
                using (var rep = RepositoryReader.Open(memoryStream, ch))
                {
                    return ModelFileUtils.LoadPipeline(env, rep, new MultiFileSource(testDataPath), true);
                }
            }
        }
    }
}
