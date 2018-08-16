// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
using Microsoft.ML.Transforms.TensorFlow;

[assembly: LoadableClass(TensorflowTransform.Summary, typeof(TensorflowTransform), typeof(TensorflowTransform.Arguments), typeof(SignatureDataTransform),
    TensorflowTransform.UserName, TensorflowTransform.ShortName)]

[assembly: LoadableClass(TensorflowTransform.Summary, typeof(TensorflowTransform), null, typeof(SignatureLoadDataTransform),
    TensorflowTransform.UserName, TensorflowTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    public sealed class TensorflowTransform : RowToRowMapperTransformBase
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

            [Argument(ArgumentType.Required, HelpText = "This is the frozen protobuf model file. Please see https://www.tensorflow.org/mobile/prepare_models for more detail(s).", ShortName = "ModelDir", SortOrder = 2)]
            public string ModelFile;
        }

        private sealed class Bindings : ManyToOneColumnBindingsBase
        {
            public sealed class TFColInfo
            {
                public readonly string[] InputColNames;
                public readonly TFShape[] TfShapes;
                public readonly TFDataType[] TfTypes;

                public TFColInfo(string[] inputColNames, TFShape[] tfShapes, TFDataType[] tfType)
                {
                    Contracts.AssertNonEmpty(tfShapes);
                    Contracts.AssertNonEmpty(tfType);
                    Contracts.Assert(tfType.Length == tfType.Length);

                    InputColNames = inputColNames;
                    TfShapes = tfShapes;
                    TfTypes = tfType;
                }
            }

            public readonly TFColInfo[] TfColInfo;
            public readonly string[] OutputColNames;
            public readonly ColumnType[] OutputCols;
            public readonly TFDataType[] OutputTFTypes;

            public Bindings(Column[] columns, ISchema schemaInput, TensorflowTransform parent)
                : base(columns, schemaInput, TestTypes)
            {
                OutputCols = new ColumnType[columns.Length];
                OutputTFTypes = new TFDataType[columns.Length];
                OutputColNames = new string[columns.Length];
                TfColInfo = new TFColInfo[columns.Length];
                for (int i=0; i<columns.Length; i++)
                {
                    OutputColNames[i] = columns[i].Name;
                    (OutputCols[i], OutputTFTypes[i]) = BuildOuputMetaData(parent._session, parent._batchSize, columns[i].Name);
                    TfColInfo[i] = BuildTFInputMetaData(parent._session, parent._batchSize, columns[i].Source);
                }
            }

            public Bindings(ModelLoadContext ctx, ISchema schema, TensorflowTransform parent)
                :base(ctx, schema, TestTypes)
            {

                int size = ctx.Reader.ReadInt32();

                OutputCols = new ColumnType[size];
                OutputTFTypes = new TFDataType[size];
                OutputColNames = new string[size];
                TfColInfo = new TFColInfo[size];
                for (int i=0;i< size; i++)
                {
                    var numCol = ctx.Reader.ReadInt32();
                    string[] source = new string[numCol];
                    for(int j=0;j<source.Length;j++)
                    {
                        source[j] = ctx.Reader.ReadString();
                    }
                    TfColInfo[i] = BuildTFInputMetaData(parent._session, parent._batchSize, source);
                }

                for (int i = 0; i < size; i++)
                {
                    var colName = ctx.Reader.ReadString();
                    OutputColNames[i] = colName;
                    (OutputCols[i], OutputTFTypes[i]) = BuildOuputMetaData(parent._session, parent._batchSize, colName);
                }
            }

            public override void Save(ModelSaveContext ctx)
            {
                base.Save(ctx);

                ctx.Writer.Write(TfColInfo.Length);
                foreach (var colInfo in TfColInfo)
                {
                    ctx.Writer.Write(colInfo.InputColNames.Length);
                    foreach (var colName in colInfo.InputColNames)
                    {
                        ctx.Writer.Write(colName);
                    }
                }

                foreach (var colName in OutputColNames)
                {
                    ctx.Writer.Write(colName);
                }
            }

            private (ColumnType, TFDataType) BuildOuputMetaData(TFSession tfSession, int batchSize, string columnName)
            {
                var tfoutput = new TFOutput(tfSession.Graph[columnName]);
                var shape = tfSession.Graph.GetTensorShape(tfoutput);

                int[] dims = new int[shape.NumDimensions];
                for (int k = 0; k < shape.NumDimensions; k++)
                {
                    dims[k] = (int)(shape[k] == -1 ? batchSize : shape[k]);
                }
                var kind = TensorflowUtils.Tf2MlNetType(tfoutput.OutputType);
               return (new VectorType(PrimitiveType.FromKind(kind), dims), tfoutput.OutputType);
            }

            private TFColInfo BuildTFInputMetaData(TFSession tfSession, int batchSize, string[] source)
            {
                var tfShapes = new TFShape[source.Length];
                var tfTypes = new TFDataType[source.Length];
                var colNames = new string[source.Length];
                for (int j = 0; j < source.Length; j++)
                {
                    colNames[j] = source[j];
                    var tfoutput = new TFOutput(tfSession.Graph[colNames[j]]);
                    tfShapes[j] = tfSession.Graph.GetTensorShape(tfoutput);
                    tfTypes[j] = tfoutput.OutputType;

                    var l = new long[tfShapes[j].NumDimensions];
                    for (int ishape = 0; ishape < tfShapes[j].NumDimensions; ishape++)
                    {
                        l[ishape] = tfShapes[j][ishape] == -1 ? batchSize : tfShapes[j][ishape];
                    }
                    tfShapes[j] = new TFShape(l);
                }
                return new TFColInfo(colNames, tfShapes, tfTypes);
            }

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo & iinfo < OutputCols.Length);

                Contracts.Assert(OutputCols[iinfo] != null);
                return OutputCols[iinfo];
            }

            private static string TestTypes(ColumnType[] types)
            {
                Contracts.AssertNonEmpty(types);
                if (!types.All(t => t.IsVector))
                    return "All source columns must be of vector type";

                if (!types.All(t => TensorflowUtils.IsTypeSupportedInTf(t)))
                    return "One of the input types is not supported in Tensorflow";

                return null;
            }
        }

        public const string Summary = "Transforms the data using the tenorflow model.";
        public const string UserName = "TensorflowTransform";
        public const string ShortName = "TFTransform";

        public const string LoaderSignature = "TFTransform";
        private const string RegistrationName = "Tensorflow";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TENSFLOW",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private readonly Bindings _bindings;

        /// <summary>
        /// Tensorflow session object
        /// </summary>
        private readonly TFSession _session;

        public override ISchema Schema => _bindings;

        /// <summary>
        ///  Any missing dimension can be a batch dimension.
        ///  Currently setting it to 1.
        /// </summary>
        private readonly int _batchSize;

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="modelFile">This is the frozen tensorflow model file. https://www.tensorflow.org/mobile/prepare_models </param>
        /// <param name="name">Name of the output column. Keep it same as in the Tensorflow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the Tensorflow model.</param>
        public TensorflowTransform(IHostEnvironment env, IDataView input, string modelFile, string name, params string[] source)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source, Name = name } }, ModelFile = modelFile }, input)
        {
        }

        public TensorflowTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));
            for (int i = 0; i < args.Column.Length; i++)
                Host.CheckUserArg(Utils.Size(args.Column[i].Source) > 0, nameof(args.Column));
            Host.CheckNonWhiteSpace(args.ModelFile, nameof(args.ModelFile));
            Host.CheckUserArg(File.Exists(args.ModelFile), nameof(args.ModelFile));

            _batchSize = 1; //  Currently setting it to 1.
            _session = LoadTFSession(args.ModelFile);
            _bindings = new Bindings(args.Column, Source.Schema, this);
        }

        private TensorflowTransform(IHost host, ModelLoadContext ctx, IDataView input)
           : base(host, input)
        {
            Host.AssertValue(ctx);

            _batchSize = ctx.Reader.ReadInt32();
#pragma warning disable MSML_NoMessagesForLoadContext
            Host.CheckDecode(_batchSize > 0, "BatchSize must be positive.");
#pragma warning restore MSML_NoMessagesForLoadContext

            byte[] data = null;
            if (!ctx.TryLoadBinaryStream("TFModel", r => data = r.ReadByteArray()))
                throw Host.ExceptDecode();

            var graph = new TFGraph();
            try
            {
                graph.Import(data);
                _session = new TFSession(graph);
            }
            catch (Exception ex)
            {
#pragma warning disable MSML_NoMessagesForLoadContext
                throw Host.ExceptDecode(ex, "Tensorflow exception triggered while loading model.");
#pragma warning restore MSML_NoMessagesForLoadContext

            }
            _bindings = new Bindings(ctx, Source.Schema, this);
        }

        ~TensorflowTransform()
        {
            _session.CloseSession();
            _session.Dispose();
        }

        public static TensorflowTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new TensorflowTransform(h, ctx, input));
        }

        private TFSession LoadTFSession(string modelFile)
        {
            var graph = new TFGraph();
            try
            {
                graph.Import(File.ReadAllBytes(modelFile), "");
            }
            catch (Exception ex)
            {
#pragma warning disable MSML_NoMessagesForLoadContext
                throw Host.ExceptDecode(ex, "Tensorflow exception triggered while loading model.");
#pragma warning restore MSML_NoMessagesForLoadContext

            }
            return new TFSession(graph);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.Writer.Write(_batchSize);

            var buffer = new TFBuffer();
            _session.Graph.ToGraphDef(buffer);

            ctx.SaveBinaryStream("TFModel", w =>
            {
                w.WriteByteArray(buffer.ToArray());
            });
            _bindings.Save(ctx);
        }

        public void DisposeTFSession()
        {
            _session.CloseSession();
            _session.DeleteSession();
        }

        private ValueGetter<T> GetSrcGetter<T>(IRow input, int iinfo, int isrc)
        {
            return input.GetGetter<T>(_bindings.Infos[iinfo].SrcIndices[isrc]);
        }

        private ITensorValueGetter CreateTensorValueGetter<T>(IRow input, ColumnType type, int colIndex, TFShape tfShape)
        {
            if (type.IsVector)
                return new TensorValueGetterVec<T>(input, colIndex, tfShape);
            else
                return new TensorValueGetter<T>(input, colIndex);
        }

        private ITensorValueGetter CreateTensorValueGetterVec(IRow input, TFDataType tfType, ColumnType columnType, int colIndex, TFShape tfShape)
        {
            var type = TFTensor.TypeFromTensorType(tfType);
            if (type != null)
            {
                return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type, input, columnType, colIndex, tfShape);
            }

            throw Host.ExceptNotSupp("Tensorflow type not supported");
        }

        private ITensorValueGetter[] GetTensorValueGetters(IRow input, int iinfo)
        {
            var info = _bindings.Infos[iinfo];
            var tfInfo = _bindings.TfColInfo[iinfo];
            var srcTensorGetters = new ITensorValueGetter[info.SrcIndices.Length];
            for (int j = 0; j < info.SrcIndices.Length; j++)
            {
                int colIndex = _bindings.Infos[iinfo].SrcIndices[j];
                srcTensorGetters[j] = CreateTensorValueGetterVec(input, tfInfo.TfTypes[j], info.SrcTypes[j], colIndex, tfInfo.TfShapes[j]);
            }
            return srcTensorGetters;
        }

        private Delegate MakeGetter(IRow input, int iinfo)
        {
            var info = _bindings.Infos[iinfo];
            var outInfo = _bindings.OutputCols[iinfo];
            var tfType = _bindings.OutputTFTypes[iinfo];
            var type = TFTensor.TypeFromTensorType(tfType);
            if (type != null)
            {
                return Utils.MarshalInvoke(MakeGetter<int>, outInfo.ItemType.RawType, input, iinfo, outInfo);
            }

            throw Host.ExceptNotSupp("Tensorflow type not supported");
        }

        private Delegate MakeGetter<T>(IRow input, int iinfo, ColumnType columnType)
        {
            Host.AssertValue(input);
            Host.Assert(typeof(T) == columnType.ItemType.RawType);

            var info = _bindings.Infos[iinfo];
            var tfInfo = _bindings.TfColInfo[iinfo];
            var srcTensorGetters = GetTensorValueGetters(input, iinfo);

            ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    var runner = _session.GetRunner();
                    for (int i = 0; i < info.SrcIndices.Length; i++)
                    {
                        var inputName = tfInfo.InputColNames[i];
                        var type = info.SrcTypes[i];
                        runner.AddInput(inputName, srcTensorGetters[i].GetTensor());
                    }

                    var tensors = runner.Fetch(_bindings.OutputColNames[iinfo]).Run();

                    Contracts.Assert(tensors.Length > 0);

                    var values = dst.Values;
                    if (Utils.Size(values) != _bindings.OutputCols[iinfo].VectorSize)
                        values = new T[_bindings.OutputCols[iinfo].VectorSize];

                    TensorflowUtils.FetchData<T>(tensors[0].Data, values);
                    dst = new VBuffer<T>(values.Length, values);
                };
            return valuegetter;
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

        private interface ITensorValueGetter
        {
            TFTensor GetTensor();
        }

        private class TensorValueGetter<T> : ITensorValueGetter
        {
            private readonly ValueGetter<T> _srcgetter;

            public TensorValueGetter(IRow input, int colIndex)
            {
                _srcgetter = input.GetGetter<T>(colIndex);
            }
            public TFTensor GetTensor()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                return TFTensor.CreateScalar(scalar);
            }
        }

        private class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private readonly TFShape _tfShape;
            private VBuffer<T> _vBuffer;
            private VBuffer<T> _vBufferDense;
            public TensorValueGetterVec(IRow input, int colIndex, TFShape tfShape)
            {
                _srcgetter = input.GetGetter<VBuffer<T>>(colIndex);
                _tfShape = tfShape;
                _vBuffer = default;
                _vBufferDense = default;
            }
            public TFTensor GetTensor()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyToDense(ref _vBufferDense);
                return TFTensor.Create(_vBufferDense.Values, _tfShape);
            }
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
