﻿// Licensed to the .NET Foundation under one or more agreements.
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
    public class PrivateAIRegressionTests
    {
        public class Encryption
        {
            public EncryptionParameters Params { get; }
            public Evaluator Evaluator { get; }
            public Encryptor Encryptor { get; }
            public Decryptor Decryptor { get; }
            public FractionalEncoder Encoder { get; }

            public Encryption()
            {
                Params = new EncryptionParameters();
                Params.PolyModulus = "1x^2048 + 1";
                Params.CoeffModulus = DefaultParams.CoeffModulus128(2048);
                Params.PlainModulus = 1 << 8;

                var context = new SEALContext(Params);

                var keygen = new KeyGenerator(context);
                var publicKey = keygen.PublicKey;
                var secretKey = keygen.SecretKey;

                Encryptor = new Encryptor(context, publicKey);
                Evaluator = new Evaluator(context);
                Decryptor = new Decryptor(context, secretKey);

                Encoder = new FractionalEncoder(context.PlainModulus, context.PolyModulus, 64, 32, 3);
            }
        }

        // Create Encryption object based on SEAL parameters
        public static Encryption EncryContext = new Encryption();

        [Fact]
        public void EncryptedScorerRegressionTest()
        {
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Train the model, get the predictor and optionally get the test IDataView
                // to test our model on.
                IDataView testData = null;
                LinearRegressionPredictor pred = TrainModel(env, ref testData);

                // Send the encryption related parameter to model so that model can be encrypted.
                // Only encryption related objects are passed to model. Decryption part is still on client side to maintain the privacy.
                pred.EncryptionContext = new EncryptionContext();
                pred.EncryptionContext.Encoder = EncryContext.Encoder;
                pred.EncryptionContext.Encryptor = EncryContext.Encryptor;
                pred.EncryptionContext.Evaluator = EncryContext.Evaluator;

                // Now encrypt the model
                pred.EncryptModel();

                // Get the valuemapper methods. Both for normal and encrypted case.
                // We will use these mappers to score the feature vector before and after encryption.
                // Since non of ML.Net transforms are encryption aware, feature vector is featurized here.
                // Featurized vector is then ecrypted and passed on to model for scoring.
                var valueMapperEncrypted = pred.GetEncruptedMapper<VBuffer<Ciphertext>, Ciphertext>();
                var valueMapper = pred.GetMapper<VBuffer<Single>, Single>();


                // Prepare for iteration over the data pipeline.
                var cursorFactory = new FloatLabelCursor.Factory(new RoleMappedData(testData, DefaultColumnNames.Label, DefaultColumnNames.Features)
                                                        , CursOpt.Label | CursOpt.Features);
                using (var cursor = cursorFactory.Create())
                {
                    // Iterate over the data and match encrypted and non-encrypted score.
                    while (cursor.MoveNext())
                    {
                        // Predict on Encrypted Data
                        var vBufferencryptedFeatures = EncryptData(ref cursor.Features);
                        Ciphertext encryptedResult = new Ciphertext();
                        valueMapperEncrypted(ref vBufferencryptedFeatures, ref encryptedResult);

                        // Decode the encrypted prediction obtrained from the model.
                        var plainResult = new Plaintext();
                        EncryContext.Decryptor.Decrypt(encryptedResult, plainResult);
                        var predictionEncrypted = (float)EncryContext.Encoder.Decode(plainResult);

                        // Predict on non-ecrypted data.
                        float prediction = 0;
                        valueMapper(ref cursor.Features, ref prediction);

                        // Compare the results to some tolerance.
                        Assert.True(Math.Abs(predictionEncrypted - prediction) <= (1e-05 + 1e-08 * Math.Abs(prediction)));
                    }
                }
            }
        }

        private VBuffer<Ciphertext> EncryptData(ref VBuffer<Single> features)
        {
            Ciphertext[] encryptedFeatures = new Ciphertext[features.Values.Length];

            for (int i = 0; i < features.Values.Length; i++)
            {
                encryptedFeatures[i] = new Ciphertext(EncryContext.Params);
                EncryContext.Encryptor.Encrypt(EncryContext.Encoder.Encode(features.Values[i]), encryptedFeatures[i]);
            }
            return new VBuffer<Ciphertext>(features.Length, features.Count, encryptedFeatures, features.Indices);
        }
        private LinearRegressionPredictor TrainModel(TlcEnvironment env, ref IDataView testData)
        {
            string dataPath = @"E:\Experiments\UCIHousing\housing.txt";
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

            // Get test data. We are testing on the same file used for training.
            testData = GetTestPipeline(env, trans, pred);
            return pred;
        }

        private IDataView GetTestPipeline(IHostEnvironment env, IDataView transforms, IPredictor pred)
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
