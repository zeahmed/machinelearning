﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using TensorFlow;

[assembly: LoadableClass(TensorflowTransform.Summary, typeof(TensorflowTransform), typeof(TensorflowTransform.Arguments), typeof(TensorflowTransform),
    TensorflowTransform.UserName, "CopyColumns", "CopyColumnsTransform", TensorflowTransform.ShortName, DocName = "transform/CopyColumnsTransform.md")]

[assembly: LoadableClass(TensorflowTransform.Summary, typeof(TensorflowTransform), null, typeof(SignatureLoadDataTransform),
    TensorflowTransform.UserName, TensorflowTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    public sealed class TensorflowTransform : RowToRowMapperTransformBase, IDisposable
    {
        public sealed class Column : ManyToOneColumn
        {
            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.Required, HelpText = "Directory where the tensorflow model is saved.", ShortName = "ModelDir", SortOrder = 2)]
            public string ModelDir;
        }

        private sealed class Bindings : ManyToOneColumnBindingsBase
        {
            public sealed class TFColInfo
            {
                public readonly TFShape[] TfShapes;
                public readonly TFDataType[] TfTypes;

                public TFColInfo(TFShape[] tfShapes, TFDataType[] tfType)
                {
                    Contracts.AssertNonEmpty(tfShapes);
                    Contracts.AssertNonEmpty(tfType);
                    Contracts.Assert(tfType.Length == tfType.Length);

                    TfShapes = tfShapes;
                    TfTypes = tfType;
                }
            }

            public readonly TFColInfo[] TfColInfo;
            public readonly string[] OutputColNames;
            public readonly ColumnType[] OutputCols;
            public Bindings(Column[] columns, ISchema schemaInput, TensorflowTransform parent)
                : base(columns, schemaInput, TestTypes)
            {
                OutputCols = new ColumnType[columns.Length];
                OutputColNames = new string[columns.Length];
                TfColInfo = new TFColInfo[columns.Length];
                for (int i=0; i<columns.Length; i++)
                {
                    OutputColNames[i] = columns[i].Name;
                    OutputCols[i] = BuildOuputMetaData(parent._session, columns[i].Name);
                    TfColInfo[i] = BuildTFInputMetaData(parent._session, columns[i].Source);
                }
            }

            private ColumnType BuildOuputMetaData(TFSession tfSession, string columnName)
            {
                var tfoutput = new TFOutput(tfSession.Graph[columnName]);
                var shape = tfSession.Graph.GetTensorShape(tfoutput);

                int[] dims = new int[shape.NumDimensions];
                for (int k = 0; k < shape.NumDimensions; k++)
                {
                    dims[k] = (int)(shape[k] == -1 ? BatchSize : shape[k]);
                }
                // Output tensor is expected to be of shape [Batch, Data]
               return new VectorType(PrimitiveType.FromKind(DataKind.R4), dims);
            }

            private TFColInfo BuildTFInputMetaData(TFSession tfSession, string[] source)
            {
                var tfShapes = new TFShape[source.Length];
                var tfTypes = new TFDataType[source.Length];
                for (int j = 0; j < source.Length; j++)
                {
                    var inputColName = source[j];
                    var tfoutput = new TFOutput(tfSession.Graph[inputColName]);
                    tfShapes[j] = tfSession.Graph.GetTensorShape(tfoutput);
                    tfTypes[j] = tfoutput.OutputType;

                    var l = new long[tfShapes[j].NumDimensions];
                    for (int ishape = 0; ishape < tfShapes[j].NumDimensions; ishape++)
                    {
                        l[ishape] = tfShapes[j][ishape] == -1 ? BatchSize : tfShapes[j][ishape];
                    }
                    tfShapes[j] = new TFShape(l);
                }
                return new TFColInfo(tfShapes, tfTypes);
            }

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < OutputCols.Length);

                Contracts.Assert(OutputCols[iinfo] != null);
                return OutputCols[iinfo];
            }

            private static string TestTypes(ColumnType[] types)
            {
                return null;
            }
        }

        public const string Summary = "Transforms the data using the tenorflow model.";
        public const string UserName = "Tensorflow Transform";
        public const string ShortName = "TFTransform";

        public const string LoaderSignature = "TFTransform";
        private const string RegistrationName = "Tensorflow";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TFTRANSFORM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private readonly Bindings _bindings;

        /// <summary>
        /// Tensorflow session object
        /// </summary>
        private TFSession _session;

        /// <summary>
        /// Tensorflow saves model with 'serve' tag when it is intended for serving.
        /// </summary>
        private const string ServingTag = "serve";

        public override ISchema Schema => _bindings;

        /// <summary>
        ///  First dimension in each tensor is a batch dimension.
        ///  Currently setting it to 1.
        /// </summary>
        private const int BatchSize = 1;

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="modelDir">Directory where the tensorflow model is saved.</param>
        /// <param name="name">Name of the output column. Keep it same as in the Tensorflow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the Tensorflow model.</param>
        public TensorflowTransform(IHostEnvironment env, IDataView input, string modelDir, string name, params string[] source)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source, Name = name } }, ModelDir = modelDir }, input)
        {
        }

        public TensorflowTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));
            for (int i = 0; i < args.Column.Length; i++)
                Host.CheckUserArg(Utils.Size(args.Column[i].Source) > 0, nameof(args.Column));

            _session = LoadTFSession(args.ModelDir);
            _bindings = new Bindings(args.Column, Source.Schema, this);
        }

        private TFSession LoadTFSession(string modelDir)
        {
            var sessionOptions = new TFSessionOptions();
            var runOptions = new TFBuffer();
            var graph = new TFGraph();
            var metaGraphDef = new TFBuffer();
            var status = new TFStatus();

            return new TFSession().FromSavedModel(
                    sessionOptions,
                    runOptions,
                    modelDir,
                    new[] { ServingTag },
                    graph,
                    metaGraphDef);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~TensorflowTransform()
        {
            Dispose(false);
        }

        public void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_session != null)
                {
                    _session.CloseSession();
                    _session.DeleteSession();
                }
                _session = null;
            }
        }

        public override void Save(ModelSaveContext ctx)
        {
            throw new NotImplementedException();
        }

        private ValueGetter<T> GetSrcGetter<T>(IRow input, int iinfo, int isrc)
        {
            return input.GetGetter<T>(_bindings.Infos[iinfo].SrcIndices[isrc]);
        }

        private float[] FetchFloatData(IntPtr data, Type type, int size)
        {
            var result = new float[size];
            Marshal.Copy(data, result, 0, size);
            return result;
        }

        private ValueGetter<VBuffer<float>> MakeGetter(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            // Need to check type here.
            //Host.Assert(Infos[iinfo].TypeSrc.IsText);

            var info = _bindings.Infos[iinfo];
            var tfInfo = _bindings.TfColInfo[iinfo];
            var srcGetterOnes = new ValueGetter<float>[info.SrcIndices.Length];
            var srcGetterVecs = new ValueGetter<VBuffer<float>>[info.SrcIndices.Length];
            for (int j = 0; j < info.SrcIndices.Length; j++)
            {
                if (info.SrcTypes[j].IsVector)
                    srcGetterVecs[j] = GetSrcGetter<VBuffer<float>>(input, iinfo, j);
                else
                    srcGetterOnes[j] = GetSrcGetter<float>(input, iinfo, j);
            }

            return (ref VBuffer<float> dst) =>
                {
                    var runner = _session.GetRunner();
                    for (int i = 0; i < info.SrcIndices.Length; i++)
                    {
                        var inputName = input.Schema.GetColumnName(info.SrcIndices[i]);
                        var type = info.SrcTypes[i];
                        if (type.IsVector)
                        {
                            VBuffer<float> tmpBuf = default;
                            srcGetterVecs[i](ref tmpBuf);
                            VBuffer<float> dense = default;
                            tmpBuf.CopyToDense(ref dense);

                            var tensor = TFTensor.FromBuffer(tfInfo.TfShapes[i], dense.Values, 0, dense.Length);
                            runner.AddInput(inputName, tensor);
                        }
                        else
                        {
                            float scalar = default;
                            srcGetterOnes[i](ref scalar);
                            runner.AddInput(inputName, new TFTensor(scalar));
                        }
                    }

                    var tensors = runner.Fetch(_bindings.OutputColNames[iinfo]).Run();

                    Contracts.Assert(tensors.Length > 0);

                    var output = FetchFloatData(tensors[0].Data, _bindings.OutputCols[iinfo].RawType, _bindings.OutputCols[iinfo].VectorSize);
                    dst = new VBuffer<float>(output.Length, output);
                };
        }

        private TFDataType GetTFType(string name)
        {
            return new TFOutput(_session.Graph[name]).OutputType;
        }

        protected override Func<int, bool> GetDependenciesCore(Func<int, bool> predicate)
        {
            return _bindings.GetDependencies(predicate);
        }

        protected override Delegate[] CreateGetters(IRow input, Func<int, bool> active, out Action disp)
        {
            Func<int, bool> activeInfos =
                iinfo =>
                {
                    int col = _bindings.MapIinfoToCol(iinfo);
                    return active(col);
                };

            var getters = new Delegate[_bindings.InfoCount];
            disp = null;
            using (var ch = Host.Start("CreateGetters"))
            {
                for (int iinfo = 0; iinfo < _bindings.InfoCount; iinfo++)
                {
                    if (!activeInfos(iinfo))
                        continue;
                    getters[iinfo] = MakeGetter(input, iinfo);
                }
                ch.Done();
                return getters;
            }
        }

        protected override int MapColumnIndex(out bool isSrc, int col)
        {
            return _bindings.MapColumnIndex(out isSrc, col);
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return true;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred, rand);
            return new RowCursor(Host, this, input, active);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
            Host.AssertNonEmpty(inputs);

            if (inputs.Length == 1 && n > 1 && _bindings.AnyNewColumnsActive(predicate))
                inputs = DataViewUtils.CreateSplitCursors(out consolidator, Host, inputs[0], n);
            Host.AssertNonEmpty(inputs);

            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new RowCursor(Host, this, inputs[i], active);
            return cursors;
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;
            private readonly Delegate[] _getters;

            public RowCursor(IChannelProvider provider, TensorflowTransform parent, IRowCursor input, bool[] active)
                : base(provider, input)
            {
                Ch.AssertValue(parent);
                Ch.Assert(active == null || active.Length == parent._bindings.ColumnCount);

                _bindings = parent._bindings;
                _active = active;

                _getters = new Delegate[_bindings.Infos.Length];
                for (int i = 0; i < _bindings.Infos.Length; i++)
                {
                    if (IsIndexActive(i))
                        _getters[i] = parent.MakeGetter(Input, i);
                }
            }

            public ISchema Schema { get { return _bindings; } }

            private bool IsIndexActive(int iinfo)
            {
                Ch.Assert(0 <= iinfo & iinfo < _bindings.Infos.Length);
                return _active == null || _active[_bindings.MapIinfoToCol(iinfo)];
            }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active == null || _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return Input.GetGetter<TValue>(index);

                Ch.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }
        }
    }
}
