package com.llama4j;

@FunctionalInterface
interface Sampler {
    Sampler ARGMAX = FloatTensor::argmax;

    int sampleToken(FloatTensor logits);
}
