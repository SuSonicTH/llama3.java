package com.llama4j;

@FunctionalInterface
interface AggregateFunction {
    float apply(float acc, float value);
}

