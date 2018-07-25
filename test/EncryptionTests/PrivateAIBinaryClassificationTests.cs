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
        public Evaluator Evaluator { get; }
        public Encryptor Encryptor { get; }
        public Decryptor Decryptor { get; }
        public FractionalEncoder Encoder { get; }

        // These keys should **NOT** be saved here
        // Its just for demo purpose
        public PublicKey PublicKey { get;  }

        public SecretKey SecretKey { get; }

        public Encryption()
        {
            var context = CreateSEALContext();
            var keygen = new KeyGenerator(context);
            PublicKey = keygen.PublicKey;
            SecretKey = keygen.SecretKey;

            Encryptor = new Encryptor(context, PublicKey);
            Evaluator = new Evaluator(context);
            Decryptor = new Decryptor(context, SecretKey);

            Encoder = new FractionalEncoder(context.PlainModulus, context.PolyModulus, 64, 32, 3);
        }

        public Encryption(PublicKey publicKey, SecretKey secretKey)
        {
            var context = CreateSEALContext();
            PublicKey = publicKey;
            SecretKey = secretKey;

            Encryptor = new Encryptor(context, publicKey);
            Evaluator = new Evaluator(context);
            Decryptor = new Decryptor(context, secretKey);

            Encoder = new FractionalEncoder(context.PlainModulus, context.PolyModulus, 64, 32, 3);
        }

        private static SEALContext CreateSEALContext()
        {
            EncryptionParameters encryptionParams = new EncryptionParameters();
            encryptionParams.PolyModulus = "1x^2048 + 1";
            encryptionParams.CoeffModulus = DefaultParams.CoeffModulus128(2048);
            encryptionParams.PlainModulus = 1 << 8;

            return new SEALContext(encryptionParams);
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
            string dataPath = @"E:\TLC_git\machinelearning\test\data\breast-cancer.txt";
            var modelFile = @"\\ct01\users\zeahmed\ML.NET\build\BreastCancer.zip";
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Train the model and save it in modelFile
                TrainBinaryModel(env, dataPath, modelFile);

                RunTest(env, dataPath, modelFile);
            }
        }

        private void RunTest(TlcEnvironment env, string dataPath, string modelFile)
        {
            // Recreate the EncryptionContext object with same keys to make sure that encryption works regardless.
            EncryContext = new Encryption(EncryContext.PublicKey, EncryContext.SecretKey);

            // Load the model
            // Set the Evaluator after the model is loaded that will be used for computing
            // Get test data. We are testing on the same file used for training.
            LinearPredictor pred = (LinearPredictor)LoadModel(env, modelFile);
            pred.Evaluator = EncryContext.Evaluator;
            IDataView testData = GetTestPipeline(env, dataPath, modelFile);

            // Get the valuemapper methods. Both for normal and encrypted case.
            // We will use these mappers to score the feature vector before and after encryption.
            // Since non of ML.Net transforms are encryption aware, feature vector is featurized here.
            // Featurized vector is then ecrypted and passed on to model for scoring.
            var valueMapperEncrypted = pred.GetEncryptedMapper<VBuffer<Ciphertext>, Ciphertext>();
            var valueMapper = pred.GetMapper<VBuffer<Single>, Single>();

            BinaryWriter writer = new BinaryWriter(new FileStream(modelFile.Replace(".zip",".bin"), FileMode.Create));
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

                    WriteData(writer, vBufferencryptedFeatures);
                }

                Output.WriteLine("Avg. Prediction Time : {0}ms", executionTime / sampleCount);
                Output.WriteLine("Avg. Prediction Time (Encrypted) : {0}ms", encryptedExecutionTime / sampleCount);
            }
            writer.Close();
        }

        private void WriteData(BinaryWriter writer, VBuffer<Ciphertext> features)
        {
            if (features.Indices == null)
            {
                writer.Write(false);
                writer.Write(features.Values.Length);
                for (int i = 0; i < features.Values.Length; i++)
                {
                    features.Values[i].Save(writer.BaseStream);
                }
            }
            else
            {
                writer.Write(true);
                writer.Write(features.Length);
                writer.Write(features.Values.Length);
                for (int i = 0; i < features.Values.Length; i++)
                {
                    writer.Write(features.Indices[i]);
                    features.Values[i].Save(writer.BaseStream);
                }
            }
        }

        private VBuffer<Ciphertext> EncryptData(ref VBuffer<Single> features)
        {
            Ciphertext[] encryptedFeatures = new Ciphertext[features.Values.Length];

            for (int i = 0; i < features.Values.Length; i++)
            {
                encryptedFeatures[i] = new Ciphertext();
                EncryContext.Encryptor.Encrypt(EncryContext.Encoder.Encode(features.Values[i]), encryptedFeatures[i]);
            }
            return new VBuffer<Ciphertext>(features.Length, features.Count, encryptedFeatures, features.Indices);
        }
        private void TrainBinaryModel(TlcEnvironment env, string dataPath, string modelFile)
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

            // Save model
            trainRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
            SaveModel(env, pred, trainRoles, modelFile);
        }

        private IDataView GetTestPipeline(IHostEnvironment env, string testDataPath, string modelFile)
        {
            using (var stream = new FileStream(modelFile, FileMode.Open))
            {
                return ModelFileUtils.LoadPipeline(env, stream, new MultiFileSource(testDataPath), true);
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
