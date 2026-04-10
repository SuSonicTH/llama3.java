package com.llama4j;

@FunctionalInterface
interface MapWithIndexFunction {
    float apply(float value, int index);
}
