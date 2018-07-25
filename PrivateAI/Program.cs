using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.Research.SEAL;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PrivateAI
{

    class Program
    {
        static void PrintUsage()
        {
            Console.Error.WriteLine("Usage:");
            Console.Error.WriteLine("PrivateAI <mode> [<EncryptedModelFile>] [<EncryptedDataFile>] [<PublicKey or PrivatKey file>]");
            Console.Error.WriteLine("where, <mode> = [-s|-e|-d|-g|-h]");
            Console.Error.WriteLine("       -m = Encrypt model (<ModelFile> and <PublicKey> are required)");
            Console.Error.WriteLine("       -s = Score using encrypted model (<EncryptedModelFile> and <EncryptedDataFile> are required)");
            Console.Error.WriteLine("       -e = Encrypt data (<EncryptedModelFile>, <EncryptedDataFile> and <PublicKey file> is required)");
            Console.Error.WriteLine("       -d = Decrypt data (<EncryptedDataFile> and <PrivatKey file> are required)");
            Console.Error.WriteLine("       -g = Generate Keys (public and private key files will be generated in the current directory");
            Console.Error.WriteLine("       -h = Help");
        }
        static void Main(string[] args)
        {
            if (args.Length == 0 || args[0] == "-h")
            {
                PrintUsage();
                return;
            }


            if (args[0] == "-m")
            {
                if (args.Length < 3)
                {
                    PrintUsage();
                    return;
                }

                var modelFile = args[1];
                var publicKeyFile = args[2];
                PrivateAIUtils.EncryptModel(modelFile, publicKeyFile);
            }
            else
            if (args[0] == "-s")
            {
                if (args.Length < 3)
                {
                    PrintUsage();
                    return;
                }

                using (var env = new TlcEnvironment(seed: 1, conc: 1))
                {
                    var dataPath = args[2];
                    var modelFile = args[1];
                    Execute(env, dataPath, modelFile);
                }
            }
            else
            if (args[0] == "-e")
            {
                if (args.Length < 4)
                {
                    PrintUsage();
                    return;
                }

                var dataPath = args[2];
                var modelFile = args[1];
                var publicKeyFile = args[3];
                PrivateAIUtils.EncryptData(args[2], args[1], publicKeyFile);
            }
            else
            if (args[0] == "-d")
            {
                if (args.Length < 3)
                {
                    PrintUsage();
                    return;
                }

                var dataPath = args[1];
                var privateKeyFile = args[2];
                PrivateAIUtils.DecryptData(dataPath, privateKeyFile);
            }
            else
            if (args[0] == "-g")
            {
                EncryptionContext context = new EncryptionContext();
                using (BinaryWriter writerPublic = new BinaryWriter(new FileStream("PublicKey", FileMode.Create)))
                using (BinaryWriter writerPrivate = new BinaryWriter(new FileStream("PrivateKey", FileMode.Create)))
                {
                    context.PublicKey.Save(writerPublic.BaseStream);
                    context.SecretKey.Save(writerPrivate.BaseStream);
                }
            }
            else
            {
                Console.Error.WriteLine("Unknown Mode: {0}", args[0]);
            }
        }
        
        private static void Execute(TlcEnvironment env, string dataPath, string modelFile)
        {
            // Recreate the EncryptionContext object with same keys to make sure that encryption works regardless.
            EncryptionContext EncryContext = new EncryptionContext();

            // Load the model
            // Set the Evaluator after the model is loaded that will be used for computing
            // Get test data. We are testing on the same file used for training.
            LinearPredictor pred = (LinearPredictor)PrivateAIUtils.LoadModel(env, modelFile);
            pred.Evaluator = EncryContext.Evaluator;

            // Get the valuemapper methods. Both for normal and encrypted case.
            // We will use these mappers to score the feature vector before and after encryption.
            // Since non of ML.Net transforms are encryption aware, feature vector is featurized here.
            // Featurized vector is then ecrypted and passed on to model for scoring.
            var valueMapperEncrypted = pred.GetEncryptedMapper<VBuffer<Ciphertext>, Ciphertext>();
            var valueMapper = pred.GetMapper<VBuffer<Single>, Single>();

            using (BinaryReader reader = new BinaryReader(new FileStream(dataPath, FileMode.Open)))
            using (BinaryWriter writer = new BinaryWriter(new FileStream(dataPath + ".out", FileMode.Create)))
            {
                double encryptedExecutionTime = 0;
                int sampleCount = 0;
                // Iterate over the data and match encrypted and non-encrypted score.
                while (reader.BaseStream.Position != reader.BaseStream.Length)
                {
                    VBuffer<Ciphertext> vBufferencryptedFeatures = ReadData(reader);
                    sampleCount++;
                    Console.WriteLine("Scoring row: {0}", sampleCount);
                    // Predict on Encrypted Data
                    Ciphertext encryptedResult = new Ciphertext();
                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    valueMapperEncrypted(ref vBufferencryptedFeatures, ref encryptedResult);
                    encryptedExecutionTime += watch.ElapsedTicks / 10000.0;
                    watch.Stop();

                    encryptedResult.Save(writer.BaseStream);
                }
                Console.WriteLine("Avg. Prediction Time : {0}ms", encryptedExecutionTime / sampleCount);
            }
        }

        private static VBuffer<Ciphertext> ReadData(BinaryReader reader)
        {
            bool isSparse = reader.ReadBoolean();

            if (isSparse)
            {
                int length = reader.ReadInt32();
                int count = reader.ReadInt32();
                int[] indices = new int[count];
                Ciphertext[] ciphertexts = new Ciphertext[count];
                for (int i = 0; i < count; i++)
                {
                    indices[i] = reader.ReadInt32();
                    ciphertexts[i].Load(reader.BaseStream);
                }
                return new VBuffer<Ciphertext>(length, count, ciphertexts, indices);
            }
            else
            {

                int length = reader.ReadInt32();
                Ciphertext[] ciphertexts = new Ciphertext[length];
                for (int i = 0; i < length; i++)
                {
                    ciphertexts[i] = new Ciphertext();
                    ciphertexts[i].Load(reader.BaseStream);
                }
                return new VBuffer<Ciphertext>(length, ciphertexts);
            }
            
        }
    }
}
