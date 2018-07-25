using Microsoft.Research.SEAL;

namespace PrivateAI
{
    public class EncryptionContext
    {
        public Evaluator Evaluator { get; }
        public Encryptor Encryptor { get; }
        public Decryptor Decryptor { get; }
        public FractionalEncoder Encoder { get; }

        // These keys should **NOT** be saved here
        // Its just for demo purpose
        public PublicKey PublicKey { get; }

        public SecretKey SecretKey { get; }

        public EncryptionContext()
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

        public EncryptionContext(PublicKey publicKey, SecretKey secretKey)
        {
            var context = CreateSEALContext();
            PublicKey = publicKey;
            SecretKey = secretKey;

            Encryptor = new Encryptor(context, publicKey);
            Evaluator = new Evaluator(context);
            Decryptor = new Decryptor(context, secretKey);

            Encoder = new FractionalEncoder(context.PlainModulus, context.PolyModulus, 64, 32, 3);
        }

        public EncryptionContext(PublicKey publicKey)
        {
            var context = CreateSEALContext();
            PublicKey = publicKey;

            Encryptor = new Encryptor(context, publicKey);
            Evaluator = new Evaluator(context);
            Decryptor = null;

            Encoder = new FractionalEncoder(context.PlainModulus, context.PolyModulus, 64, 32, 3);
        }

        public EncryptionContext(SecretKey secretKey)
        {
            var context = CreateSEALContext();
            SecretKey = secretKey;

            Encryptor = null;
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
}
