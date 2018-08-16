// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
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

namespace Microsoft.ML.Transforms.TensorFlow
{
    internal partial class TensorflowUtils
    {
        internal static DataKind Tf2MlNetType(TFDataType type)
        {
            switch (type)
            {
                case TFDataType.Float:
                    return DataKind.R4;
                case TFDataType.Double:
                    return DataKind.R8;
                case TFDataType.Int32:
                    return DataKind.I4;
                case TFDataType.Int64:
                    return DataKind.I8;
                case TFDataType.UInt32:
                    return DataKind.U4;
                case TFDataType.UInt64:
                    return DataKind.U8;
                case TFDataType.Bool:
                    return DataKind.Bool;
                case TFDataType.String:
                    return DataKind.TX;
                default:
                    throw new NotSupportedException("Tensorflow type not supported.");
            }
        }

        internal static bool IsTypeSupportedInTf(ColumnType type)
        {
            try
            {
                if (type.IsVector)
                {
                    TFTensor.TensorTypeFromType(type.ItemType.RawType);
                    return true;
                }

                TFTensor.TensorTypeFromType(type.RawType);
                return true;
            }
           catch (ArgumentOutOfRangeException)
            {
                return false;
            }
        }

        public static unsafe T[] FetchData<T>(IntPtr data, int size)
        {
            var result = new T[size];

            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr target = handle.AddrOfPinnedObject();

            Int64 sizeInBytes = size * Marshal.SizeOf((typeof(T)));
            Buffer.MemoryCopy(data.ToPointer(), target.ToPointer(), sizeInBytes, sizeInBytes);
            handle.Free();
            return result;
        }

        public static unsafe void FetchData<T>(IntPtr data, T[] result)
        {
            var size = result.Length;

            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr target = handle.AddrOfPinnedObject();

            Int64 sizeInBytes = size * Marshal.SizeOf((typeof(T)));
            Buffer.MemoryCopy(data.ToPointer(), target.ToPointer(), sizeInBytes, sizeInBytes);
            handle.Free();
        }
    }
}
