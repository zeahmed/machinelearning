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
using Microsoft.ML.Runtime.Internal.Calibration;
using System.Threading;
using System.Globalization;
using Xunit.Abstractions;

namespace EncryptionTests
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

    public partial class PrivateAITests
    {
        private readonly string _dataRoot;

        public PrivateAITests(ITestOutputHelper output)
        {
            //This locale is currently set for tests only so that the produced output
            //files can be compared on systems with other locales to give set of known
            //correct results that are on en-US locale.
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");

            var currentAssemblyLocation = new FileInfo(typeof(PrivateAITests).Assembly.Location);
            var _rootDir = currentAssemblyLocation.Directory.Parent.Parent.Parent.Parent.FullName;
            var _outDir = Path.Combine(currentAssemblyLocation.Directory.FullName, "TestOutput");
            Directory.CreateDirectory(_outDir);
            _dataRoot = Path.Combine(_rootDir, "test", "data");
            Output = output;
        }

        protected ITestOutputHelper Output { get; }

        protected string GetDataPath(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(_dataRoot, name));
        }

        // Create Encryption object based on SEAL parameters
        public Encryption EncryContext = new Encryption();

        [Fact]
        public void EncryptedScorerBinaryClassificationTest()
        {
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Train the model, get the predictor and optionally get the test IDataView
                // to test our model on.
                // Have to remove the calibirator because Sigmoid and other complex functions are not supported by SEAL
                IDataView testData = null;
                LinearBinaryPredictor pred = TrainBinaryModel(env, ref testData);

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
                        Assert.True(Math.Abs(predictionEncrypted - prediction) <= (1e-03 + 1e-08 * Math.Abs(prediction)));
                    }

                    Output.WriteLine("Avg. Prediction Time : {0}ms", executionTime / sampleCount);
                    Output.WriteLine("Avg. Prediction Time (Encrypted) : {0}ms", encryptedExecutionTime / sampleCount);
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
        private LinearBinaryPredictor TrainBinaryModel(TlcEnvironment env, ref IDataView testData)
        {
            string dataPath = @"E:\TLC_git\machinelearning\test\data\breast-cancer.txt";
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
                                Source = new [] { new TextLoader.Range() { Min = 1, Max = 9} },
                                Type = DataKind.R4
                            }
                    }
                }, new MultiFileSource(dataPath));

            // Normalizer is not automatically added though the trainer has 'NormalizeFeatures' On/Auto
            var trans = NormalizeTransform.CreateMinMaxNormalizer(env, loader, "Features");

            // Train
            var trainer = new LogisticRegression(env, new LogisticRegression.Arguments() { NumThreads = 1 });
            // Explicity adding CacheDataView since caching is not working though trainer has 'Caching' On/Auto
            var cached = new CacheDataView(env, trans, prefetch: null);
            var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");

            // Have to remove the calibirator because Sigmoid and other complex functions are not supported by SEAL
            var pred = (LinearBinaryPredictor)trainer.Train(trainRoles).SubPredictor;

            // Set the Evaluator that will be used for computing
            pred.Evaluator = EncryContext.Evaluator;

            // Now encrypt the model
            pred.EncryptModel(EncryContext.Encryptor, EncryContext.Encoder);

            var modelFile = @"\\ct01\users\zeahmed\ML.NET\build\BreastCancer.zip";
            // Save model
            SaveModel(env, pred, trainRoles, modelFile);
            pred = (LinearBinaryPredictor)LoadModel(env, modelFile);

            // Set the Evaluator after the model is loaded that will be used for computing
            pred.Evaluator = EncryContext.Evaluator;

            // Get test data. We are testing on the same file used for training.
            testData = GetTestPipelineBinary(env, trans, pred);
            return pred;
        }

        private IDataView GetTestPipelineBinary(IHostEnvironment env, IDataView transforms, IPredictor pred)
        {
            string testDataPath = @"E:\TLC_git\machinelearning\test\data\breast-cancer.txt";
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

        private void SaveModel(IHostEnvironment env, IPredictor pred, RoleMappedData trainRoles, string modelFile)
        {
            using (var ch = env.Start("Saving model"))
            using (var filestream = new FileStream(modelFile, FileMode.Create))
            {
                // Model cannot be saved with CacheDataView
                TrainUtils.SaveModel(env, ch, filestream, pred, trainRoles);
            }
        }

        private IPredictor LoadModel(IHostEnvironment env, string modelFile)
        {
            using (var filestream = new FileStream(modelFile, FileMode.Open))
            {
                // Model cannot be saved with CacheDataView
                return ModelFileUtils.LoadPredictorOrNull(env, filestream);
            }
        }
    }
}
