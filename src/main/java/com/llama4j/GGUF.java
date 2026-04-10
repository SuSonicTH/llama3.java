package com.llama4j;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class GGUF {
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    // Buffered read window for GGUF header + metadata + tensor-info parsing.
    // This avoids many tiny FileChannel.read() calls.
    private static final int PARSE_BUFFER_SIZE = 1 << 20;
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(3);
    private int magic;
    private int version;
    private int tensorCount; // uint64_t
    private int alignment;
    private int metadata_kv_count; // uint64_t
    private Map<String, Object> metadata;

    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return tensorInfos;
    }

    private Map<String, GGUFTensorInfo> tensorInfos;

    private long tensorDataOffset;

    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    private final ByteBuffer BB_1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_4 = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private long parsePosition;

    public static GGUF loadModel(FileChannel fileChannel, String modelLabel) throws IOException {
        try (var ignored = Timer.log("Parse " + modelLabel)) {
            // A caller may reuse an already-open channel after previous reads; always parse from start.
            fileChannel.position(0L);
            GGUF gguf = new GGUF();
            ReadableByteChannel channel = Channels.newChannel(
                    new BufferedInputStream(Channels.newInputStream(fileChannel), PARSE_BUFFER_SIZE)
            );
            gguf.parsePosition = 0L;
            gguf.loadModelImpl(channel);
            return gguf;
        }
    }

    enum MetadataValueType {
        // The value is a 8-bit unsigned integer.
        // (represented as signed byte in Java)
        UINT8(1),
        // The value is a 8-bit signed integer.
        INT8(1),
        // The value is a 16-bit unsigned little-endian integer.
        UINT16(2),
        // The value is a 16-bit signed little-endian integer.
        INT16(2),
        // The value is a 32-bit unsigned little-endian integer.
        UINT32(4),
        // The value is a 32-bit signed little-endian integer.
        INT32(4),
        // The value is a 32-bit IEEE754 floating point number.
        FLOAT32(4),
        // The value is a boolean.
        // 1-byte value where 0 is false and 1 is true.
        // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
        // (0 = false, 1 = true)
        BOOL(1),
        // The value is a UTF-8 non-null-terminated string, with length prepended.
        // Variable-size payload (length-prefixed), therefore byteSize is marked as -8.
        STRING(-8),
        // The value is an array of other values, with element type and length prepended.
        // Arrays can be nested.
        // Variable-size payload, therefore byteSize is marked as -8.
        ARRAY(-8),
        // The value is a 64-bit unsigned little-endian integer.
        UINT64(8),
        // The value is a 64-bit signed little-endian integer.
        INT64(8),
        // The value is a 64-bit IEEE754 floating point number.
        FLOAT64(8);

        private final int byteSize;

        MetadataValueType(int byteSize) {
            this.byteSize = byteSize;
        }

        private static final MetadataValueType[] VALUES = values();

        public static MetadataValueType fromIndex(int index) {
            return VALUES[index];
        }

        public int byteSize() {
            return byteSize;
        }
    }

    private void loadModelImpl(ReadableByteChannel channel) throws IOException {
        // The header of the file.
        readHeader(channel); // gguf_header_t header;

        // Tensor infos, which can be used to locate the tensor data.
        // gguf_tensor_info_t tensor_infos[header.tensor_count];
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUFTensorInfo ti = readTensorInfo(channel);
            assert !tensorInfos.containsKey(ti.name);
            tensorInfos.put(ti.name, ti);
        }

        // Padding to the nearest multiple of `ALIGNMENT`.
        // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
        long position = parsePosition;
        int padding = (int) ((getAlignment() - (position % getAlignment())) % getAlignment());
        skipBytes(channel, padding);

        // Tensor data.
        //
        // This is arbitrary binary data corresponding to the weights of the model. This data should be close
        // or identical to the data in the original model file, but may be different due to quantization or
        // other optimizations for inference. Any such deviations should be recorded in the metadata or as
        // part of the architecture definition.
        //
        // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
        // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
        // should be padded to `ALIGNMENT` bytes.
        // uint8_t tensor_data[];
        this.tensorDataOffset = parsePosition;
    }

    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset, Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
        // Global arena keeps tensor mmap slices alive for the process lifetime.
        // (Arena.ofAuto() can release too early when references outlive parsing scope.)
        Arena arena = Arena.global();
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());
        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();
            long numberOfElements = FloatTensor.numberOfElementsLong(ti.dimensions());
            long sizeInBytes = ti.ggmlType().byteSizeFor(numberOfElements);
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
        return tensorEntries;
    }

    public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {
    }

    private GGMLType readGGMLType(ReadableByteChannel channel) throws IOException {
        int ggmlTypeId = readInt(channel);
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUFTensorInfo readTensorInfo(ReadableByteChannel channel) throws IOException {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        // gguf_string_t name;
        String name = readString(channel);
        assert name.length() <= 64;

        // The number of dimensions in the tensor.
        // Currently at most 4, but this may change in the future.
        // uint32_t n_dimensions;
        int n_dimensions = readInt(channel);
        assert n_dimensions <= 4;

        // The dimensions of the tensor.
        int[] dimensions = new int[n_dimensions]; // uint64_t dimensions[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(channel));
        }

        // The type of the tensor.
        GGMLType ggmlType = readGGMLType(channel); // ggml_type type;

        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // The offset is relative to the tensor-data section, not file start.
        // Must be a multiple of `ALIGNMENT`.
        long offset = readLong(channel); // uint64_t offset;
        assert offset % getAlignment() == 0;
        return new GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(ReadableByteChannel channel) throws IOException {
        // A string in GGUF.
        // The length of the string, in bytes.
        int len = Math.toIntExact(readLong(channel)); // uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        // char string[len];
        return new String(readBytes(channel, len), StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(ReadableByteChannel channel) throws IOException {
        // The key of the metadata. It is a standard GGUF string, with the following caveats:
        // - It must be a valid ASCII string.
        // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
        // - It must be at most 2^16-1/65535 bytes long.
        // Any keys that do not follow these rules are invalid.
        // The key of the metadata. It must be hierarchical lower_snake_case segments separated by '.'.
        String key = readString(channel); // gguf_string_t key;
        assert key.length() < (1 << 16);
        assert key.codePoints().allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = readMetadataValue(channel);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(ReadableByteChannel channel) throws IOException {
        // The type of the value.
        // Must be one of the gguf_metadata_value_type values.
        MetadataValueType valueType = readMetadataValueType(channel); // gguf_metadata_value_type value_type;

        // The value payload itself.
        return readMetadataValueOfType(valueType, channel); // gguf_metadata_value_t value;
    }

    void readHeader(ReadableByteChannel channel) throws IOException {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        // Some readers may check for 0x46554747 on little-endian machines.
        this.magic = readInt(channel); // uint32_t magic;
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }

        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update the metadata
        // to signify the change.
        // Must be 3 for the latest format in the GGUF spec.
        // This version should only be increased for structural file-format changes.
        this.version = readInt(channel); // uint32_t version;
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }

        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is always present
        // for loading the tensors.
        // This is explicit to make tensor loading possible without metadata lookup.
        this.tensorCount = Math.toIntExact(readLong(channel)); // uint64_t tensor_count;

        // The number of metadata key-value pairs.
        this.metadata_kv_count = Math.toIntExact(readLong(channel)); // uint64_t metadata_kv_count;

        // The metadata key-value pairs.
        this.metadata = HashMap.newHashMap(metadata_kv_count); // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(channel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    private Object readArray(ReadableByteChannel channel) throws IOException {
        // Any value type is valid, including arrays.
        // Any metadata value type is valid, including nested arrays.
        MetadataValueType valueType = readMetadataValueType(channel); // gguf_metadata_value_type type;

        // Number of elements, not bytes
        int len = Math.toIntExact(readLong(channel)); // uint64_t len;

        // The array of values.
        // gguf_metadata_value_t array[len];
        switch (valueType) {
            case UINT8, INT8 -> {
                return readBytes(channel, len);
            }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(channel);
                }
                return shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(channel);
                }
                return ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(channel);
                }
                return floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(channel);
                }
                return booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(channel);
                }
                return strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) {
                    arrays[i] = readArray(channel);
                }
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + valueType);
        }
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, ReadableByteChannel channel) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(channel);
            case UINT16, INT16 -> readShort(channel);
            case UINT32, INT32 -> readInt(channel);
            case FLOAT32 -> readFloat(channel);
            case UINT64, INT64 -> readLong(channel);
            case FLOAT64 -> readDouble(channel);
            case BOOL -> readBoolean(channel);
            case STRING -> readString(channel);
            case ARRAY -> readArray(channel);
        };
    }

    private MetadataValueType readMetadataValueType(ReadableByteChannel channel) throws IOException {
        int index = readInt(channel);
        return MetadataValueType.fromIndex(index);
    }

    private byte[] readBytes(ReadableByteChannel channel, int length) throws IOException {
        byte[] bytes = new byte[length];
        readFully(channel, ByteBuffer.wrap(bytes));
        return bytes;
    }

    private void skipBytes(ReadableByteChannel channel, int length) throws IOException {
        int remaining = length;
        byte[] scratch = new byte[Math.min(length, 1 << 12)];
        while (remaining > 0) {
            int chunk = Math.min(remaining, scratch.length);
            readFully(channel, ByteBuffer.wrap(scratch, 0, chunk));
            remaining -= chunk;
        }
    }

    private void readFully(ReadableByteChannel channel, ByteBuffer byteBuffer) throws IOException {
        int expected = byteBuffer.remaining();
        while (byteBuffer.hasRemaining()) {
            int bytesRead = channel.read(byteBuffer);
            if (bytesRead < 0) {
                throw new IOException("Unexpected EOF while reading GGUF metadata");
            }
        }
        parsePosition += expected;
    }

    private byte readByte(ReadableByteChannel channel) throws IOException {
        BB_1.clear();
        readFully(channel, BB_1);
        return BB_1.get(0);
    }

    private boolean readBoolean(ReadableByteChannel channel) throws IOException {
        return readByte(channel) != 0;
    }

    private short readShort(ReadableByteChannel channel) throws IOException {
        BB_2.clear();
        readFully(channel, BB_2);
        return BB_2.getShort(0);
    }

    private int readInt(ReadableByteChannel channel) throws IOException {
        BB_4.clear();
        readFully(channel, BB_4);
        return BB_4.getInt(0);
    }

    private long readLong(ReadableByteChannel channel) throws IOException {
        BB_8.clear();
        readFully(channel, BB_8);
        return BB_8.getLong(0);
    }

    private float readFloat(ReadableByteChannel channel) throws IOException {
        return Float.intBitsToFloat(readInt(channel));
    }

    private double readDouble(ReadableByteChannel channel) throws IOException {
        return Double.longBitsToDouble(readLong(channel));
    }

    public int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}
