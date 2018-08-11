// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        private class TestData
        {
            public float[] a;
            public float[] b;
        }
        [Fact]
        public void TensorflowTransformMatrixMultiplicationTest()
        {
            var model_location = GetDataPath("model_matmul");
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = ComponentCreation.CreateDataView(env,
                    new List<TestData>(new TestData[] { new TestData() { a = new[] { 1.0f, 2.0f,
                                                                                     3.0f, 4.0f },
                                                                         b = new[] { 1.0f, 2.0f,
                                                                                     3.0f, 4.0f } },
                        new TestData() { a = new[] { 2.0f, 2.0f,
                                                     2.0f, 2.0f },
                                         b = new[] { 3.0f, 3.0f,
                                                     3.0f, 3.0f } } }));

                var trans = new TensorflowTransform(env, loader, model_location, "c", "a", "b");

                using (var cursor = trans.GetRowCursor(a => true))
                {
                    var cgetter = cursor.GetGetter<VBuffer<float>>(2);
                    Assert.True(cursor.MoveNext());
                    VBuffer<float> c = default;
                    cgetter(ref c);

                    Assert.Equal(1.0 * 1.0 + 2.0 * 3.0, c.Values[0]);
                    Assert.Equal(1.0 * 2.0 + 2.0 * 4.0, c.Values[1]);
                    Assert.Equal(3.0 * 1.0 + 4.0 * 3.0, c.Values[2]);
                    Assert.Equal(3.0 * 2.0 + 4.0 * 4.0, c.Values[3]);

                    Assert.True(cursor.MoveNext());
                    c = default;
                    cgetter(ref c);

                    Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, c.Values[0]);
                    Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, c.Values[1]);
                    Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, c.Values[2]);
                    Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, c.Values[3]);

                    Assert.False(cursor.MoveNext());

                }
            }
        }

        [Fact]
        public void TensorflowTransformMNISTConvTest()
        {
            var model_location = GetDataPath("mnist_model");
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                var dataPath = GetDataPath("mnist_train.1K.tsv");
                var testDataPath = GetDataPath("mnist_test.1K.tsv");

                // Pipeline
                var loader = new TextLoader(env,
                new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoader.Range() { Min=0, Max=0} },
                            Type = DataKind.Num
                        },

                        new TextLoader.Column()
                        {
                            Name = "Placeholder",
                            Source = new [] { new TextLoader.Range() { Min=1, Max=784} },
                            Type = DataKind.Num
                        }
                    }
                }, new MultiFileSource(dataPath));

                IDataView trans = new TensorflowTransform(env, loader, model_location, "Softmax", "Placeholder");
                trans = new ConcatTransform(env, trans, "reshape_input", "Placeholder");
                trans = new TensorflowTransform(env, trans, model_location, "dense/Relu", "reshape_input");
                trans = new ConcatTransform(env, trans, "Features", "Softmax", "dense/Relu");

                var trainer = new LightGbmMulticlassTrainer(env, new LightGbmArguments());

                 var cached = new CacheDataView(env, trans, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                var pred = trainer.Train(trainRoles);
            }
        }
    }
}
