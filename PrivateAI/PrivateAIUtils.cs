using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using Microsoft.Research.SEAL;
using System;
using System.IO;

namespace PrivateAI
{
    public class PrivateAIUtils
    {
        public static void EncryptModel(string modelFile, string publicKeyFile)
        {
            PublicKey key = new PublicKey();
            using (BinaryReader reader = new BinaryReader(new FileStream(publicKeyFile, FileMode.Open)))
            {
                key.Load(reader.BaseStream);
            }

            EncryptionContext context = new EncryptionContext(key);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                LinearPredictor pred = (LinearPredictor)LoadModel(env, modelFile);
                pred.Evaluator = context.Evaluator;

                // Now encrypt the model
                pred.EncryptModel(context.Encryptor, context.Encoder);

                // Save model
                var trainRoles = LoadRoleMapping(env, modelFile);
                SaveModel(env, pred, trainRoles, modelFile + ".encrypted");
            }
        }

        public static void EncryptData(string dataPath, string modelFile, string publicKeyFile)
        {
            PublicKey key = new PublicKey();
            using (BinaryReader reader = new BinaryReader(new FileStream(publicKeyFile, FileMode.Open)))
            {
                key.Load(reader.BaseStream);
            }

            EncryptionContext context = new EncryptionContext(key);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                IDataView testData = GetTestPipeline(env, dataPath, modelFile);
                LinearPredictor pred = (LinearPredictor)LoadModel(env, modelFile);
                pred.Evaluator = context.Evaluator;

                // Get the valuemapper methods. Both for normal and encrypted case.
                // We will use these mappers to score the feature vector before and after encryption.
                // Since non of ML.Net transforms are encryption aware, feature vector is featurized here.
                // Featurized vector is then ecrypted and passed on to model for scoring.
                var valueMapperEncrypted = pred.GetEncryptedMapper<VBuffer<Ciphertext>, Ciphertext>();
                var valueMapper = pred.GetMapper<VBuffer<Single>, Single>();

                BinaryWriter writer = new BinaryWriter(new FileStream(dataPath + ".encrypted", FileMode.Create));
                // Prepare for iteration over the data pipeline.
                var cursorFactory = new FloatLabelCursor.Factory(new RoleMappedData(testData, DefaultColumnNames.Label, DefaultColumnNames.Features)
                                                        , CursOpt.Label | CursOpt.Features);
                using (var cursor = cursorFactory.Create())
                {
                    int sampleCount = 0;
                    // Iterate over the data and match encrypted and non-encrypted score.
                    while (cursor.MoveNext())
                    {
                        sampleCount++;
                        // Predict on Encrypted Data
                        var vBufferencryptedFeatures = EncryptData(context, ref cursor.Features);
                        Ciphertext encryptedResult = new Ciphertext();
                        var watch = System.Diagnostics.Stopwatch.StartNew();
                        valueMapperEncrypted(ref vBufferencryptedFeatures, ref encryptedResult);


                        WriteData(writer, vBufferencryptedFeatures);
                    }
                }
            }
        }

        public static void DecryptData(string dataPath, string privateKeyFile)
        {
            SecretKey key = new SecretKey();
            using (BinaryReader reader = new BinaryReader(new FileStream(privateKeyFile, FileMode.Open)))
            {
                key.Load(reader.BaseStream);
            }

            EncryptionContext context = new EncryptionContext(key);

            using (BinaryReader reader = new BinaryReader(new FileStream(dataPath, FileMode.Open)))
            {
                while (reader.BaseStream.Position != reader.BaseStream.Length)
                {
                    var ciphertext = new Ciphertext();
                    ciphertext.Load(reader.BaseStream);

                    var plainResult = new Plaintext();
                    context.Decryptor.Decrypt(ciphertext, plainResult);
                    var predictionEncrypted = (float)context.Encoder.Decode(plainResult);
                    Console.WriteLine(predictionEncrypted);
                }
            }
        }

        public static void WriteData(BinaryWriter writer, VBuffer<Ciphertext> features)
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

        public static VBuffer<Ciphertext> EncryptData(EncryptionContext EncryContext, ref VBuffer<Single> features)
        {
            Ciphertext[] encryptedFeatures = new Ciphertext[features.Values.Length];

            for (int i = 0; i < features.Values.Length; i++)
            {
                encryptedFeatures[i] = new Ciphertext();
                EncryContext.Encryptor.Encrypt(EncryContext.Encoder.Encode(features.Values[i]), encryptedFeatures[i]);
            }
            return new VBuffer<Ciphertext>(features.Length, features.Count, encryptedFeatures, features.Indices);
        }

        public static IDataView GetTestPipeline(IHostEnvironment env, string testDataPath, string modelFile)
        {
            using (var stream = new FileStream(modelFile, FileMode.Open))
            {
                return ModelFileUtils.LoadPipeline(env, stream, new MultiFileSource(testDataPath), true);
            }
        }

        public static void SaveModel(IHostEnvironment env, IPredictor pred, RoleMappedData trainRoles, string modelFile)
        {
            using (var ch = env.Start("Saving model"))
            using (var filestream = new FileStream(modelFile, FileMode.Create))
            {
                // Model cannot be saved with CacheDataView
                TrainUtils.SaveModel(env, ch, filestream, pred, trainRoles);
            }
        }

        public static IPredictor LoadModel(IHostEnvironment env, string modelFile)
        {
            using (var filestream = new FileStream(modelFile, FileMode.Open))
            {
                // Model cannot be saved with CacheDataView
                return ModelFileUtils.LoadPredictorOrNull(env, filestream);
            }
        }

        public static RoleMappedData LoadRoleMapping(IHostEnvironment env, string modelFile)
        {
            using (var filestream = new FileStream(modelFile, FileMode.Open))
            {
                var dataview = ModelFileUtils.LoadPipeline(env, filestream, new MultiFileSource(null), true);
                // Model cannot be saved with CacheDataView
                return new RoleMappedData(dataview, ModelFileUtils.LoadRoleMappingsOrNull(env, filestream));
            }
        }
    }
}
