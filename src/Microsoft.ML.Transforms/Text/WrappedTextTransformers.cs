﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Data.DataLoadSave;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Transforms.Text
{

    /// <summary>
    /// Produces a bag of counts of ngrams (sequences of consecutive words) in a given text.
    /// It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.
    /// </summary>
    public sealed class WordBagEstimator : IEstimator<ITransformer>
    {
        private readonly IHost _host;
        private readonly (string outputColumnName, string[] sourceColumnsNames)[] _columns;
        private readonly int _ngramLength;
        private readonly int _skipLength;
        private readonly bool _allLengths;
        private readonly int _maxNumTerms;
        private readonly NgramExtractingEstimator.WeightingCriteria _weighting;

        /// <summary>
        /// Options for how the ngrams are extracted.
        /// </summary>
        public class Options
        {
            /// <summary>
            /// Maximum ngram length.
            /// </summary>
            public int NgramLength;

            /// <summary>
            /// Maximum number of tokens to skip when constructing an ngram.
            /// </summary>
            public int SkipLength;

            /// <summary>
            /// Whether to store all ngram lengths up to ngramLength, or only ngramLength.
            /// </summary>
            public bool AllLengths;

            /// <summary>
            /// The maximum number of grams to store in the dictionary, for each level of ngrams,
            /// from 1 (in position 0) up to ngramLength (in position ngramLength-1)
            /// </summary>
            public int[] MaxNumTerms;

            /// <summary>
            /// The weighting criteria.
            /// </summary>
            public NgramExtractingEstimator.WeightingCriteria Weighting;

            public Options()
            {
                NgramLength = 1;
                SkipLength = NgramExtractingEstimator.Defaults.SkipLength;
                AllLengths = NgramExtractingEstimator.Defaults.AllLengths;
                MaxNumTerms = new int[] { NgramExtractingEstimator.Defaults.MaxNumTerms };
                Weighting = NgramExtractingEstimator.Defaults.Weighting;
            }
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumnName"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        internal WordBagEstimator(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            : this(env, outputColumnName, new[] { inputColumnName ?? outputColumnName }, ngramLength, skipLength, allLengths, maxNumTerms, weighting)
        {
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumnNames"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">The column containing output tokens.</param>
        /// <param name="inputColumnNames">The columns containing text to compute bag of word vector.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        internal WordBagEstimator(IHostEnvironment env,
            string outputColumnName,
            string[] inputColumnNames,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            : this(env, new[] { (outputColumnName, inputColumnNames) }, ngramLength, skipLength, allLengths, maxNumTerms, weighting)
        {
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        internal WordBagEstimator(IHostEnvironment env,
            (string outputColumnName, string[] inputColumnNames)[] columns,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(WordBagEstimator));

            foreach (var (outputColumnName, inputColumnName) in columns)
            {
                _host.CheckUserArg(Utils.Size(inputColumnName) > 0, nameof(columns));
                _host.CheckValue(outputColumnName, nameof(columns));
            }

            _columns = columns;
            _ngramLength = ngramLength;
            _skipLength = skipLength;
            _allLengths = allLengths;
            _maxNumTerms = maxNumTerms;
            _weighting = weighting;
        }

        /// <summary> Trains and returns a <see cref="ITransformer"/>.</summary>
        public ITransformer Fit(IDataView input)
        {
            // Create arguments.
            var options = new WordBagBuildingTransformer.Options
            {
                Columns = _columns.Select(x => new WordBagBuildingTransformer.Column { Name = x.outputColumnName, Source = x.sourceColumnsNames }).ToArray(),
                NgramLength = _ngramLength,
                SkipLength = _skipLength,
                AllLengths = _allLengths,
                MaxNumTerms = new[] { _maxNumTerms },
                Weighting = _weighting
            };

            return new TransformWrapper(_host, WordBagBuildingTransformer.Create(_host, options, input), true);
        }

        /// <summary>
        /// Schema propagation for estimators.
        /// Returns the output schema shape of the estimator, if the input schema shape is like the one provided.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            var fakeSchema = FakeSchemaFactory.Create(inputSchema);
            var transformer = Fit(new EmptyDataView(_host, fakeSchema));
            return SchemaShape.Create(transformer.GetOutputSchema(fakeSchema));
        }
    }

    /// <summary>
    /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text.
    /// It does so by hashing each ngram and using the hash value as the index in the bag.
    /// </summary>
    public sealed class WordHashBagEstimator : IEstimator<ITransformer>
    {
        private readonly IHost _host;
        private readonly (string outputColumnName, string[] inputColumnNames)[] _columns;
        private readonly int _hashBits;
        private readonly int _ngramLength;
        private readonly int _skipLength;
        private readonly bool _allLengths;
        private readonly uint _seed;
        private readonly bool _ordered;
        private readonly int _invertHash;

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumnName"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">The column containing bag of word vector. Null means <paramref name="inputColumnName"/> is replaced.</param>
        /// <param name="inputColumnName">The column containing text to compute bag of word vector.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        internal WordHashBagEstimator(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : this(env, new[] { (outputColumnName, new[] { inputColumnName ?? outputColumnName }) }, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash)
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumnNames"/>
        /// and outputs bag of word vector as <paramref name="outputColumnName"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">The column containing output tokens.</param>
        /// <param name="inputColumnNames">The columns containing text to compute bag of word vector.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        internal WordHashBagEstimator(IHostEnvironment env,
            string outputColumnName,
            string[] inputColumnNames,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : this(env, new[] { (outputColumnName, inputColumnNames) }, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash)
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        internal WordHashBagEstimator(IHostEnvironment env,
            (string outputColumnName, string[] inputColumnNames)[] columns,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(WordHashBagEstimator));

            foreach (var (input, output) in columns)
            {
                _host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
                _host.CheckValue(output, nameof(input));
            }

            _columns = columns;
            _hashBits = hashBits;
            _ngramLength = ngramLength;
            _skipLength = skipLength;
            _allLengths = allLengths;
            _seed = seed;
            _ordered = ordered;
            _invertHash = invertHash;
        }

        /// <summary> Trains and returns a <see cref="ITransformer"/>.</summary>
        public ITransformer Fit(IDataView input)
        {
            // Create arguments.
            var options = new WordHashBagProducingTransformer.Options
            {
                Columns = _columns.Select(x => new WordHashBagProducingTransformer.Column { Name = x.outputColumnName  ,Source = x.inputColumnNames}).ToArray(),
                HashBits = _hashBits,
                NgramLength = _ngramLength,
                SkipLength = _skipLength,
                AllLengths = _allLengths,
                Seed = _seed,
                Ordered = _ordered,
                InvertHash = _invertHash
            };

            return new TransformWrapper(_host, WordHashBagProducingTransformer.Create(_host, options, input), true);
        }

        /// <summary>
        /// Schema propagation for estimators.
        /// Returns the output schema shape of the estimator, if the input schema shape is like the one provided.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            var fakeSchema = FakeSchemaFactory.Create(inputSchema);
            var transformer = Fit(new EmptyDataView(_host, fakeSchema));
            return SchemaShape.Create(transformer.GetOutputSchema(fakeSchema));
        }
    }
}