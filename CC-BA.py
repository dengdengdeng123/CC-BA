def channel_max(x):
    x=K.max(x,axis=-1,keepdims=True)
    return x
def channel_avg(x):
    x=K.mean(x,axis=-1,keepdims=True)
    return x
def se_block(inputs,ratio=4):
    x_max=Lambda(channel_max)(inputs)
    x_avg=Lambda(channel_avg)(inputs)
    x=Concatenate(axis=2)([x_max,x_avg])
    x=Conv1D(1,kernel_size=7,padding='same',activation='sigmoid')(x)
    return Multiply()([inputs,x])

def channel_block(inputs,ratio=4):
    channel = inputs.shape[-1]

    share_dense1=Dense(channel//4)
    share_dense2=Dense(channel)

    x_max = GlobalMaxPooling1D()(inputs)
    x_avg = GlobalAveragePooling1D()(inputs)

    x_max = Reshape([1, -1])(x_max)
    x_avg = Reshape([1, -1])(x_avg)

    x_max = share_dense1(x_max)
    x_max = share_dense2(x_max)

    x_avg = share_dense1(x_avg)
    x_avg = share_dense2(x_avg)

    x = Add()([x_max,x_avg])
    x = Activation('sigmoid')(x)

    out = Multiply()([inputs,x])
    return out

inputs = Input()
conv1_1 = Conv1D(name='conv1_1', filters=32, kernel_size=3, activation='relu', padding="same")(inputs)
dp1_1 = Dropout(0.5)(conv1_1)
conv1_2 = Conv1D(name='conv1_2', filters=32, strides=5, kernel_size=3, activation='relu', padding="same")(dp1_1)
conv1_3 = Conv1D(name='conv1_3', filters=32, kernel_size=3, activation='relu', padding="same")(conv1_2)
dp1_3 = Dropout(0.5)(conv1_3)
conv1_4 = Conv1D(name='conv1_4', filters=32, strides=5, kernel_size=3, activation='relu', padding="same")(dp1_3)
convs.append(conv1_4)

conv3_1 = Conv1D(name='conv3_1', filters=32, kernel_size=5, activation='relu', padding="same")(inputs)
dp3_1 = Dropout(0.5)(conv3_1)
conv3_2 = Conv1D(name='conv3_2', filters=32, strides=5, kernel_size=5, activation='relu', padding="same")(dp3_1)
conv3_3 = Conv1D(name='conv3_3', filters=32, kernel_size=3, activation='relu', padding="same")(conv3_2)
dp3_3 = Dropout(0.5)(conv3_3)
conv3_4 = Conv1D(name='conv3_4', filters=32, strides=5, kernel_size=3, activation='relu', padding="same")(dp3_3)
convs.append(conv3_4)

conv4_1 = Conv1D(name='conv4_1', filters=32, kernel_size=7, activation='relu', padding="same")(inputs)
dp4_1 = Dropout(0.5)(conv4_1)
conv4_2 = Conv1D(name='conv4_2', filters=32, strides=5, kernel_size=7, activation='relu', padding="same")(dp4_1)
conv4_3 = Conv1D(name='conv4_3', filters=32, kernel_size=3, activation='relu', padding="same")(conv4_2)
dp4_3 = Dropout(0.5)(conv4_3)
conv4_4 = Conv1D(name='conv4_4', filters=32, strides=5, kernel_size=3, activation='relu', padding="same")(dp4_3)
convs.append(conv4_4)
merge = keras.layers.concatenate(convs, axis=2)

z2 = channel_block(merge)
z2 = se_block(z2)

conv5_1 = Conv1D(name='conv5_1', filters=64, kernel_size=3, activation='relu', padding="same")(z2)
conv5_2 = Conv1D(name='conv5_2', filters=64, kernel_size=3, strides=5, activation='relu', padding="same")(conv5_1)
f1 = Flatten()(conv5_2)
fc1 = Dense(128)(f1)
cnn_feature = Flatten(name='Flatten1')(fc1)

convs2.append(cnn_feature)
inputs2 = Embedding(256, 64, input_length=784)(inputs)
inputs3 = Reshape((784, 64))(inputs2)
conv2_1 = Bidirectional(GRU(32, dropout=0.5, return_sequences=True))(inputs3)
conv2_2 = Bidirectional(GRU(64, dropout=0.5))(conv2_1)

shape = conv2_2.shape[-1]
x = Dense(shape, activation='softmax', name="atten")(conv2_2)
z = Multiply()([conv2_2, x])

convs2.append(z)
for conv in convs2:
    print(conv)
merge2 = 0.6*convs2[0]+0.4*convs2[1]
shape = merge2.shape[-1]

output = Dense(10, activation='softmax')(merge2)
model = Model(inputs=inputs, outputs=output)
