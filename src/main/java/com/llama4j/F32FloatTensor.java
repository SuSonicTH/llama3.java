package com.llama4j;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

final class F32FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public F32FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    private static float vectorDot(F32FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            FloatVector thizVector = FloatVector.fromMemorySegment(F_SPECIES, thiz.memorySegment, (thisOffset + i) * (long) Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
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
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromMemorySegment(species, memorySegment, index * (long) Float.BYTES, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        return readFloat(memorySegment, index * (long) Float.BYTES);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }
}
