package com.llama4j;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

final class Q5_KFloatTensor extends FloatTensor {

    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q5_K.getTypeSize();

    final int size;
    final MemorySegment memorySegment;

    public Q5_KFloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q5_K;
    }

    @Override
    public float getFloat(long index) {
        // Q5_K extends Q4_K by adding one high bit per quant in qh[32].
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES);
        long scalesOffset = blockOffset + 2L * GGMLType.FLOAT16_BYTES;
        long qhOffset = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12;
        long qsOffset = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12 + 32;

        // Same 4 x 64 grouping strategy as Q4_K.
        int group = withinBlock / 64;
        int inGroup = withinBlock % 64;
        boolean isHigh = inGroup >= 32;
        int l = isHigh ? inGroup - 32 : inGroup;
        int subBlock = isHigh ? group * 2 + 1 : group * 2;

        int sc = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32L + l);
        int nibble = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);

        int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
        int qhBit = (Byte.toUnsignedInt(readByte(memorySegment, qhOffset + l)) >> qhBitPos) & 1;

        int quant = nibble | (qhBit << 4);
        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q5_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        // Dual accumulators improve instruction-level parallelism.
        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES);
            long scalesOff = blockOffset + 2L * GGMLType.FLOAT16_BYTES;
            long qhOff = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12;
            long qsOff = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12 + 32;
            var qh0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff, ByteOrder.LITTLE_ENDIAN);
            var qh1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff + 16, ByteOrder.LITTLE_ENDIAN);

            for (int g = 0; g < 4; g++) {
                int loSubBlock = g * 2;
                int hiSubBlock = loSubBlock + 1;
                float d1 = d * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, false);
                float m1 = dmin * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, true);
                float d2 = d * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, false);
                float m2 = dmin * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, true);
                int qhBitPosLo = 2 * g;
                int qhBitPosHi = qhBitPosLo + 1;
                long groupQsOff = qsOff + (long) g * 32;
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, -m1);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, -m2);

                for (int c = 0; c < 2; c++) {
                    int loBase = thatOffset + j + g * 64 + c * 16;
                    int hiBase = thatOffset + j + g * 64 + 32 + c * 16;

                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            groupQsOff + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var loQ = wBytes.and((byte) 0xF);
                    var hiQ = wBytes.lanewise(VectorOperators.LSHR, 4);

                    // Merge the 5th quantization bit from qh into low/high nibble lanes.
                    var qhBytes = c == 0 ? qh0 : qh1;
                    loQ = loQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hiQ = hiQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQf = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                        }
                        case 256 -> {
                            var loQf0 = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQf1 = loQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            var hiQf0 = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf1 = hiQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQf0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val = loQf1.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase + F_SPECIES.length()), val);
                            val2 = hiQf0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                            val2 = hiQf1.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var loQf = loQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var hiQf = hiQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase + off), val);
                                val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase + off), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }

        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}
