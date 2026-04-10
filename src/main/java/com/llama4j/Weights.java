package com.llama4j;

import java.nio.FloatBuffer;

/**
 * @param token_embedding_table token embedding table (vocab_size, dim)
 * @param rms_att_weight        weights for rmsnorms (layer, dim) rmsnorm weights
 * @param wq                    weights for matmuls (layer, n_heads * head_size)
 * @param wk                    (layer, n_kv_heads, head_size)
 * @param wv                    (layer, n_kv_heads * head_size)
 * @param wo                    (layer, n_heads * head_size, dim)
 * @param rms_ffn_weight        (layer, dim)
 * @param w1                    weights for ffn (layer, hidden_dim, dim)
 * @param w2                    (layer, dim, hidden_dim)
 * @param w3                    (layer, hidden_dim, dim)
 * @param rms_final_weight      public final rmsnorm (dim,)
 * @param freq_cis_real         freq_cis for RoPE relatively positional embeddings (seq_len, head_size/2)
 * @param freq_cis_imag         (seq_len, head_size/2)
 * @param wcls                  (optional) classifier weights for the logits, on the last layer (vocab_size, dim)
 */
public record Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo, FloatBuffer[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
}
