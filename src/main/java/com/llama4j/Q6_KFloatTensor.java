package com.llama4j;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

final class Q6_KFloatTensor extends FloatTensor {

    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q6_K.getTypeSize();

    final int size;
    final MemorySegment memorySegment;

    public Q6_KFloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    private static float vectorDot(Q6_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        // Separate accumulators for alternating stripes.
        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            long qlOff = blockOffset;
            long qhOff = blockOffset + 128;
            long scOff = blockOffset + 192;
            // Avoid Float.float16ToFloat intrinsic here: Graal Native Image can miscompile this
            // specific hot loop. Use the local non-intrinsic conversion helper instead.
            float d = float16ToFloat(readShort(thiz.memorySegment, blockOffset + 192 + 16));

            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64L;
                long qhBase = qhOff + h * 32L;

                int base = thatOffset + j + h * 128;
                for (int c = 0; c < 2; c++) {
                    var qlA = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qlB = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qhV = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN);

                    // Reconstruct signed 6-bit quants from ql nibbles + qh two-bit planes.
                    var q0 = qlA.and((byte) 0xF).or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q1 = qlB.and((byte) 0xF).or(qhV.lanewise(VectorOperators.LSHR, 2).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q2 = qlA.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 4).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q3 = qlB.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 6).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);

                    float ds0 = d * readByte(thiz.memorySegment, scOff + h * 8L + c);
                    float ds1 = d * readByte(thiz.memorySegment, scOff + h * 8L + 2 + c);
                    float ds2 = d * readByte(thiz.memorySegment, scOff + h * 8L + 4 + c);
                    float ds3 = d * readByte(thiz.memorySegment, scOff + h * 8L + 6 + c);

                    var ds0Vec = FloatVector.broadcast(F_SPECIES, ds0);
                    var ds1Vec = FloatVector.broadcast(F_SPECIES, ds1);
                    var ds2Vec = FloatVector.broadcast(F_SPECIES, ds2);
                    var ds3Vec = FloatVector.broadcast(F_SPECIES, ds3);

                    int sg0Idx = base + c * 16;
                    int sg1Idx = base + 32 + c * 16;
                    int sg2Idx = base + 64 + c * 16;
                    int sg3Idx = base + 96 + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var q0f = q0.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx), val);
                            var q1f = q1.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx), val2);
                            var q2f = q2.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx), val);
                            var q3f = q3.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx), val2);
                        }
                        case 256 -> {
                            for (int p = 0; p < 2; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), val);
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), val2);
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), val);
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), val2);
                            }
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), val);
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), val2);
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), val);
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), val2);
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
        return GGMLType.Q6_K;
    }

    @Override
    public float getFloat(long index) {
        // Q6_K block layout (256 values):
        // - ql[128]: low 4 bits for two halves
        // - qh[64]: two extra bits per quant
        // - scales[16]: signed int8 per 16-value stripe
        // - d (fp16): shared block scale
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        long qlOff = blockOffset;
        long qhOff = blockOffset + 128;
        long scOff = blockOffset + 192;
        float d = readFloat16(memorySegment, blockOffset + 192 + 16);

        // Two 128-value halves per block.
        int half = withinBlock / 128;
        int rem128 = withinBlock % 128;
        int sub32 = rem128 / 32;
        int l = rem128 % 32;

        long qlBase = qlOff + half * 64L;
        long qhBase = qhOff + half * 32L;

        int qlNibble;
        int qhShift;
        switch (sub32) {
            case 0 -> {
                qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) & 0xF;
                qhShift = 0;
            }
            case 1 -> {
                qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) & 0xF;
                qhShift = 2;
            }
            case 2 -> {
                qlNibble = (Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) >> 4) & 0xF;
                qhShift = 4;
            }
            case 3 -> {
                qlNibble = (Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) >> 4) & 0xF;
                qhShift = 6;
            }
            default -> throw new IllegalStateException();
        }

        int qhBits = (Byte.toUnsignedInt(readByte(memorySegment, qhBase + l)) >> qhShift) & 3;
        int q6 = (qlNibble | (qhBits << 4)) - 32;
        int sc = readByte(memorySegment, scOff + half * 8L + sub32 * 2L + l / 16); // signed int8

        return d * sc * q6;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }
}
