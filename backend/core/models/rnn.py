import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import tensorflow as tf2
import numpy as np
import math

from backend.utils.config import cfg
from backend.utils.util import LoadClasses
from backend.core.models.cnn import FV
from backend.core.models.cnn import NFeatures

Classes = LoadClasses(cfg.CHAR_LIST)
NClasses = len(Classes)

def RNN(Inputs, SeqLens, Scope):
    with tf.variable_scope(Scope):
        ################################################################
        # Construct batch sequences for LSTM (Giữ nguyên logic của bạn)
        maxLen = tf.reduce_max(SeqLens, 0)

        n = 0; offset = 0
        ndxs = tf.reshape(tf.range(offset, SeqLens[n] + offset), [SeqLens[n], 1])
        res = tf.gather_nd(Inputs, [ndxs])
        res = tf.reshape(res, [-1])
        zero_padding = tf.zeros([NFeatures * maxLen] - tf.shape(res), dtype=res.dtype)
        a_padded = tf.concat([res, zero_padding], 0)
        result = tf.reshape(a_padded, [maxLen, NFeatures])
        Inputs2 = result

        for n in range(1, cfg.BatchSize):
            offset = tf.cumsum(SeqLens)[n-1]
            ndxs = tf.reshape(tf.range(offset, SeqLens[n]+offset), [SeqLens[n], 1])
            res = tf.gather_nd(Inputs, [ndxs])
            res = tf.reshape(res, [-1])
            zero_padding = tf.zeros([NFeatures * maxLen] - tf.shape(res), dtype=res.dtype)
            a_padded = tf.concat([res, zero_padding], 0)
            result = tf.reshape(a_padded, [maxLen, NFeatures])
            Inputs2 = tf.concat([Inputs2, result], 0)

        n = 0
        ndxs = tf.reshape(tf.range(n, cfg.BatchSize * maxLen, maxLen), [cfg.BatchSize, 1])
        Inputs = tf.gather_nd(Inputs2, [ndxs])

        i = tf.constant(1)
        def condition(i, prev): return tf.less(i, maxLen)

        def body(i, prev):
            ndxs = tf.reshape(tf.range(i, cfg.BatchSize * maxLen, maxLen), [cfg.BatchSize, 1])
            result = tf.gather_nd(Inputs2, [ndxs])
            next = tf.concat([prev, result], 0)
            return [tf.add(i, 1), next]

        i, Inputs = tf.while_loop(condition, body, [i, Inputs], 
                                  shape_invariants=[i.get_shape(), tf.TensorShape([None, cfg.BatchSize, NFeatures])])
        
       ###############################################################
        # Construct LSTM layers - FIX LỖI TIME_MAJOR TRÊN KERAS 3

        initializer = tf2.initializers.GlorotUniform()

        # 1. Định nghĩa các Cells
        stacked_rnn_forward = []
        for i in range(cfg.NLayers):
            stacked_rnn_forward.append(tf.keras.layers.LSTMCell(units=cfg.NUnits, implementation=1))
        forward_cell = tf.keras.layers.StackedRNNCells(stacked_rnn_forward)

        stacked_rnn_backward = []
        for i in range(cfg.NLayers):
            stacked_rnn_backward.append(tf.keras.layers.LSTMCell(units=cfg.NUnits, implementation=1))
        backward_cell = tf.keras.layers.StackedRNNCells(stacked_rnn_backward)

        # 2. Xử lý chiều dữ liệu: Chuyển từ [Time, Batch, Features] sang [Batch, Time, Features]
        # Vì Keras 3 RNN mặc định chạy Batch-major
        Inputs_BM = tf.transpose(Inputs, [1, 0, 2])

        # 3. Khai báo RNN Layer (Đã bỏ time_major=True để tránh lỗi)
        fw_layer = tf.keras.layers.RNN(forward_cell, return_sequences=True)
        bw_layer = tf.keras.layers.RNN(backward_cell, return_sequences=True, go_backwards=True)

        # 4. Chạy Layer trên dữ liệu đã transpose
        fw_out_BM = fw_layer(Inputs_BM)
        bw_out_BM = bw_layer(Inputs_BM)

        # 5. Chuyển kết quả ngược lại về Time-major [Time, Batch, Units] để khớp với code phía dưới
        fw_out_seq = tf.transpose(fw_out_BM, [1, 0, 2])
        bw_out_seq = tf.transpose(bw_out_BM, [1, 0, 2])

        # Đảo ngược chuỗi Backward để khớp với thứ tự chuỗi Forward
        bw_out_seq = tf.reverse(bw_out_seq, axis=[0]) 

        # 6. Reshaping và phần còn lại giữ nguyên
        fw_out = tf.reshape(fw_out_seq, [-1, cfg.NUnits])
        bw_out = tf.reshape(bw_out_seq, [-1, cfg.NUnits])

        # 4. Các tham số Linear Layer (Giữ nguyên từ code gốc của bạn)
        W_fw = tf.Variable(tf.truncated_normal(shape=[cfg.NUnits, NClasses], stddev=np.sqrt(2.0 / cfg.NUnits), dtype=tf.float32), dtype=tf.float32)
        W_bw = tf.Variable(tf.truncated_normal(shape=[cfg.NUnits, NClasses], stddev=np.sqrt(2.0 / cfg.NUnits), dtype=tf.float32), dtype=tf.float32)
        b_out = tf.constant(0.1, shape=[NClasses], dtype=tf.float32)

        # 5. Thực hiện phép toán Affine: logits = (fw_out * W_fw) + (bw_out * W_bw) + b_out
        logits = tf.add(tf.add(tf.matmul(fw_out, W_fw), tf.matmul(bw_out, W_bw)), b_out)
        
        return tf.reshape(logits, [-1, cfg.BatchSize, NClasses])